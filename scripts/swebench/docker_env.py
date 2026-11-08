"""Run an instance's Python environment in Docker when uv cannot provide it.

Python 3.5-3.7 has no arm64 build, which rules out about 29% of SWE-bench Verified
(113 of 231 Django instances among them). For those, the official prebuilt instance
image already holds the exact environment, so we use it.

The agent's shell runs under a Seatbelt profile that denies all network access,
including Docker's socket, and we do not widen it: a Docker socket would let the agent
mount any host folder. Instead a trusted broker on the host serves the agent:

    agent  ->  ./.swe/run 'pytest -x tests/foo.py'
                 writes  .swe/req/<id>.cmd          (a file inside the project folder)
    broker ->  docker exec <fixed container> ...    (host side, never the agent)
                 writes  .swe/res/<id>.out / .rc
    agent  <-  prints the output, exits with the exit code

The container has no network, bounded memory/CPU, and bind-mounts only the checkout at
/testbed. The broker runs nothing on the host except `docker exec` into that container.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from env import EnvUnavailable, Prepared, sh

IMAGE_NAMESPACE = "swebench"   # "swebench" = the official images, "logicstar" = the compressed bundle
ALLOW_PULL = True              # False in offline mode: an image that is not already local is an error
MEMORY = "6g"
CPUS = "4"
EXEC_TIMEOUT = 900          # seconds per command inside the container
MAX_OUTPUT = 200_000        # bytes kept from one command
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate testbed"

# What the agent calls. Pure file exchange, so it works inside the network-less sandbox.
RUN_SCRIPT = r'''#!/bin/bash
# Run a command inside this instance's Docker environment, in /testbed (this folder).
#   ./.swe/run 'python -m pytest tests/test_x.py -x -q'
d="$(cd "$(dirname "$0")" && pwd)"
id="$$-$RANDOM-$RANDOM"
mkdir -p "$d/req" "$d/res"
printf '%s' "$*" > "$d/req/$id.tmp"
mv "$d/req/$id.tmp" "$d/req/$id.cmd"
waited=0
while [ ! -f "$d/res/$id.rc" ]; do
  sleep 0.3
  waited=$((waited + 1))
  if [ "$waited" -gt 6000 ]; then echo "[swe] no answer from the environment broker" >&2; exit 124; fi
done
cat "$d/res/$id.out"
rc="$(cat "$d/res/$id.rc")"
rm -f "$d/res/$id.out" "$d/res/$id.rc"
exit "${rc:-1}"
'''


def configure(namespace: str | None = None, allow_pull: bool | None = None) -> None:
    """Choose the image set and whether missing images may be pulled."""
    global IMAGE_NAMESPACE, ALLOW_PULL
    if namespace:
        IMAGE_NAMESPACE = namespace
    if allow_pull is not None:
        ALLOW_PULL = allow_pull


def image_name(instance_id: str, namespace: str | None = None) -> str:
    return f"{namespace or IMAGE_NAMESPACE}/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def image_present(name: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


def docker_ready() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def install_runner(root: Path) -> None:
    """Create <root>/.swe with the run script and its exchange folders."""
    swe = root / ".swe"
    for sub in ("req", "res", "scratch"):
        (swe / sub).mkdir(parents=True, exist_ok=True)
    run = swe / "run"
    run.write_text(RUN_SCRIPT)
    run.chmod(0o755)


def prepare_docker(instance: dict, work: Path, test_cmd: str, notes: list[str]) -> Prepared:
    """Copy the repo out of the prebuilt image and start the environment container."""
    if not docker_ready():
        raise EnvUnavailable("docker is not running; this instance needs Python 3.5-3.7")
    iid = instance["instance_id"]
    img = image_name(iid)
    tag = uuid.uuid4().hex[:6]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", iid)
    ctr, src = f"swe-{safe}-{tag}", f"swe-src-{safe}-{tag}"
    dest = work / "checkouts" / iid
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    # An image that is already local (a loaded bundle, or one kept earlier) is used as it is.
    # Only what we pull ourselves is removed afterwards.
    pulled = False
    if not image_present(img):
        if not ALLOW_PULL:
            raise EnvUnavailable(f"image {img} is not available locally and pulling is disabled")
        pull = subprocess.run(["docker", "pull", "--platform", "linux/amd64", "-q", img],
                              capture_output=True, text=True)
        if pull.returncode != 0:
            raise EnvUnavailable(f"could not pull {img}: {(pull.stderr or pull.stdout)[-200:]}")
        pulled = True
    try:
        dest.mkdir(parents=True)
        # The built tree (including compiled extensions) comes out of the image as it is.
        sh(f"docker create --platform linux/amd64 --name {src} {img}")
        sh(f"docker cp {src}:/testbed/. {dest}/", timeout=1800)
    finally:
        subprocess.run(["docker", "rm", "-f", src], capture_output=True)
    # The image's repo carries full history, including the upstream fix. Drop it.
    shutil.rmtree(dest / ".git", ignore_errors=True)
    sh("git init -q && git add -A && "
       "git -c user.name=bench -c user.email=bench@example.com commit -q -m base", cwd=dest)
    (dest / ".git" / "info" / "exclude").write_text(".swe/\n__pycache__/\n*.pyc\n")
    install_runner(dest)
    base_sha = sh("git rev-parse HEAD", cwd=dest).stdout.strip()

    started = subprocess.run(
        ["docker", "run", "-d", "--platform", "linux/amd64", "--name", ctr, "--network", "none",
         "--memory", MEMORY, "--cpus", CPUS, "--pids-limit", "2048",
         "-v", f"{dest}:/testbed", "-w", "/testbed", img, "sleep", "infinity"],
        capture_output=True, text=True)
    if started.returncode != 0:
        if pulled:
            cleanup_image(img)
        raise EnvUnavailable(f"could not start the environment container: {started.stderr[-200:]}")
    notes.append(f"environment runs in Docker ({img}{'' if pulled else ', already local'})")
    return Prepared(dest, base_sha, test_cmd, notes, mode="docker", container=ctr, image=img, pulled=pulled)


async def _exec(container: str, cmd: str, timeout: int = EXEC_TIMEOUT) -> tuple[str, int]:
    inner = f"{CONDA_ACTIVATE} && export PYTHONDONTWRITEBYTECODE=1 && {cmd}"
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "-w", "/testbed", container,
        "timeout", str(timeout), "bash", "-lc", inner,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    data, _ = await proc.communicate()
    text = data.decode(errors="replace")
    if len(text) > MAX_OUTPUT:
        text = "[output truncated]\n" + text[-MAX_OUTPUT:]
    return text, proc.returncode if proc.returncode is not None else 1


async def serve(root: Path, container: str, stop: asyncio.Event) -> None:
    """Answer the agent's requests, one at a time, until told to stop."""
    swe = root / ".swe"
    while not stop.is_set():
        for cmd_file in sorted((swe / "req").glob("*.cmd")):
            claimed = cmd_file.with_suffix(".run")
            try:
                cmd_file.rename(claimed)
            except OSError:
                continue
            out, rc = await _exec(container, claimed.read_text())
            rid = cmd_file.stem
            (swe / "res" / f"{rid}.out").write_text(out)
            (swe / "res" / f"{rid}.rc").write_text(str(rc))     # last: the script waits on it
            claimed.unlink(missing_ok=True)
        await asyncio.sleep(0.25)


@contextlib.asynccontextmanager
async def broker(prepared: Prepared):
    """Serve the agent for the duration of its run. A no-op for host (uv) environments."""
    if prepared.mode != "docker":
        yield
        return
    stop = asyncio.Event()
    task = asyncio.create_task(serve(prepared.repo_dir, prepared.container, stop))
    try:
        yield
    finally:
        stop.set()
        await task


def cleanup_image(image: str) -> None:
    subprocess.run(["docker", "rmi", "-f", image], capture_output=True)


def cleanup(prepared: Prepared, keep_image: bool = False) -> None:
    if prepared.mode != "docker":
        return
    subprocess.run(["docker", "rm", "-f", prepared.container], capture_output=True)
    if prepared.pulled and not keep_image and prepared.image:
        cleanup_image(prepared.image)


def task_text(test_cmd: str, folder: str) -> str:
    """The environment paragraph of the agent's task, for a Docker-backed instance.

    ``folder`` is the name the project is mapped under (the agent's file tools see it as
    /links/<folder>/). write_file to /output/ lands outside the container, so scratch
    scripts must be written under the project instead.
    """
    return (
        "Python environment: the project and its dependencies are installed in a Docker "
        "container, not on this Mac, so host python cannot import the project. Run every "
        "python or test command through `./.swe/run '<command>'`; it runs inside the "
        "container at /testbed, which is this same folder. Paths differ by tool: your file "
        f"tools (read_file, edit_file, grep, write_file) use /links/{folder}/..., while "
        "commands run through `./.swe/run` see the same files as /testbed/... or relative "
        "paths. /links does not exist inside the container and /testbed does not exist "
        "outside it, so never mix them. Run `./.swe/run` from the folder root. Example: "
        "`./.swe/run 'python -m pytest tests/test_x.py -x -q'`. Project test command: "
        f"`{test_cmd}` (append the test files or names you want, and run it through "
        "`./.swe/run`).\n"
        "Scratch scripts: write them with write_file to "
        f"`/links/{folder}/.swe/scratch/<name>.py` (NOT /output/, which the container cannot "
        "see) and run them with `./.swe/run 'python .swe/scratch/<name>.py'`. Files in "
        ".swe/ are not part of your patch. There is no network.\n\n"
    )
