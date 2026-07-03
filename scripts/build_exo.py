#!/usr/bin/env python3
"""Build a pruned, relocatable, prebuilt exo runtime artifact (CI / build-time).

This does once, on a CI runner, the work that
``backend.exo_cli.provision_exo`` otherwise does on every user's machine:
clone exo at a pinned ref, ``uv sync`` (which compiles the Rust/pyo3
extension and installs the custom MLX fork exo pins for distributed
RDMA/GPU-lock fixes), and build the dashboard. It then *prunes* everything
not needed to serve inference and packs the result into a single
``.tar.gz`` + ``.sha256`` that Otto downloads on demand (see
``backend.exo_runtime``).

The artifact is **Apple-Silicon only** — exo pins an arm64-only MLX fork,
and Otto only ships/notarizes a macOS arm64 build.

Pipeline
--------
1. clone exo @ ref            (git, with tarball fallback)
2. ``uv sync``                (Rust ext + python deps + MLX fork)
3. dashboard ``npm`` build    (static assets exo serves)
4. prune                      (drop build-only / dev / unused-heavy deps)
5. make relocatable           (deref interpreter, relative shebangs)
6. pack + checksum            (exo-runtime-<ref>-<arch>.tar.gz)

NOTE: steps 5/6 (relocatability + notarization) must be validated on a real
macOS CI runner — see ``.github/workflows/build-app.yml``. Codesigning and
notarization of the Mach-O files inside the venv are handled by the CI
workflow / ``scripts/sign_and_notarize.sh`` after this script produces the
tree, before packing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist" / "exo"

DEFAULT_EXO_REPO_URL = "https://github.com/exo-explore/exo.git"
DEFAULT_EXO_REF = "v1.0.71"

# Packages that exist in exo's venv only for build/dev/optional features and
# are NOT needed to serve LLM inference on Apple Silicon. Mirrors the
# philosophy of ``_exclude_args`` in scripts/build_backend.py.
#
# exo's own source imports none of these directly (verified by grep); they
# are transitive/optional. MLX is the inference path. Removing them takes the
# runtime from ~3.1 GB to ~600 MB.
PRUNE_SITE_PACKAGES = [
    # Build-only: node shipped as a wheel purely to build the dashboard, and
    # the dashboard's own node_modules. We keep only the built static output.
    "nodejs_wheel",
    # Dev tooling.
    "basedpyright",
    # Heavy frameworks pulled in transitively but unused for MLX serving —
    # the same set Otto already excludes from its backend bundle.
    "torch",
    "torchvision",
    "torchaudio",
    "pyarrow",
    "cv2",
    "sympy",
    "pandas",
    "matplotlib",
    "fontTools",
    "scipy",
    "datasets",
]


def arch_tag() -> str:
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{arch}-apple-darwin"


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[build_exo] $ {' '.join(args)}", flush=True)
    subprocess.run(list(args), cwd=str(cwd) if cwd else None, env=env, check=True)


def clone_exo(repo_url: str, ref: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--depth", "1", "--branch", ref, repo_url, str(dest))


def uv_sync(repo: Path) -> None:
    env = os.environ.copy()
    # Corporate TLS proxies: let uv use the system trust store.
    env.setdefault("UV_NATIVE_TLS", "1")
    # Force a uv-managed (portable, python-build-standalone) interpreter.
    # If left unset, uv's default "managed" preference will happily fall
    # back to a pre-installed system interpreter — e.g. a Homebrew
    # ``python@3.13`` — when one already satisfies the pinned version.
    # Homebrew's Python is a Framework build whose binaries hard-code an
    # ABSOLUTE ``/opt/homebrew/Cellar/python@X.Y/.../Python.framework``
    # load path (no ``@rpath``), which ``make_relocatable()`` below has no
    # way to bundle. Shipping that interpreter produces an artifact that
    # only runs on the exact build machine and dies with a dyld
    # "Library not loaded" error on every other Mac. ``only-managed``
    # guarantees uv always downloads/uses its own relocatable, shared-
    # library CPython instead.
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    run("uv", "sync", cwd=repo, env=env)


def build_dashboard(repo: Path) -> None:
    dash = repo / "dashboard"
    if not dash.exists():
        print("[build_exo] no dashboard/ — skipping")
        return
    run("npm", "ci", cwd=dash)
    run("npm", "run", "build", cwd=dash)


def _site_packages(repo: Path) -> Path | None:
    libs = sorted((repo / ".venv" / "lib").glob("python*/site-packages"))
    return libs[0] if libs else None


def prune(repo: Path) -> None:
    """Delete build-only / dev / unused-heavy packages and caches."""
    sp = _site_packages(repo)
    if sp is None:
        raise RuntimeError("could not locate .venv site-packages to prune")

    for name in PRUNE_SITE_PACKAGES:
        for path in list(sp.glob(f"{name}")) + list(sp.glob(f"{name}-*.dist-info")) \
                + list(sp.glob(f"{name}.*")) + list(sp.glob(f"{name}.libs")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
            print(f"[build_exo] pruned {path.relative_to(repo)}")

    # Dashboard build deps (keep only the built static output).
    node_modules = repo / "dashboard" / "node_modules"
    if node_modules.exists():
        shutil.rmtree(node_modules, ignore_errors=True)
        print("[build_exo] pruned dashboard/node_modules")

    # Drop bytecode caches and bundled test trees to shave more weight.
    removed = 0
    for cache in repo.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    print(f"[build_exo] removed {removed} __pycache__ dirs")

    # The .git history is irrelevant to a runtime artifact.
    git_dir = repo / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)


def relativize_pth(repo: Path) -> None:
    """Rewrite absolute in-repo ``.pth`` entries to be relative to site-packages.

    ``uv sync`` installs exo itself as an *editable* package: instead of a real
    ``site-packages/exo/`` it drops an ``exo.pth`` containing the ABSOLUTE path
    to the source tree on the build host (``.../dist/exo/src/src``). That path
    doesn't exist on a user's machine, so the unpacked artifact dies with
    ``ModuleNotFoundError: No module named 'exo'`` even though the source travels
    inside the tarball.

    ``site`` resolves a relative ``.pth`` line against the site-packages dir
    (``site.makepath(sitedir, line)``), so converting in-repo absolute entries to
    paths relative to site-packages makes them resolve wherever the artifact is
    unpacked. Lines that execute code (``import ...``) or point outside the repo
    are left untouched.
    """
    sp = _site_packages(repo)
    if sp is None:
        raise RuntimeError("could not locate .venv site-packages to relativize .pth")
    repo_resolved = repo.resolve()
    for pth in sp.glob("*.pth"):
        lines = pth.read_text().splitlines()
        out: list[str] = []
        changed = False
        for line in lines:
            s = line.strip()
            if s and not s.startswith("import ") and os.path.isabs(s):
                try:
                    under_repo = Path(s).resolve().is_relative_to(repo_resolved)
                except (OSError, ValueError):
                    under_repo = False
                if under_repo:
                    rel = os.path.relpath(s, sp)
                    out.append(rel)
                    changed = True
                    print(f"[build_exo] relativized {pth.name}: {s} -> {rel}")
                    continue
            out.append(line)
        if changed:
            pth.write_text("\n".join(out) + "\n")


def make_relocatable(repo: Path) -> None:
    """Best-effort: rewrite venv console scripts to a relative interpreter.

    A venv normally hard-codes an absolute interpreter path in every
    ``bin/`` script shebang and symlinks ``bin/python`` to a uv-managed
    interpreter. For the artifact to run from any user's app-data path we:

    1. dereference ``bin/python*`` symlinks so the real interpreter travels
       inside the artifact, and
    2. rewrite each script shebang to a ``/bin/sh`` polyglot that resolves
       the interpreter relative to the script's own directory.

    Validate end-to-end on CI; if uv gains a clean retro-relocate this can
    be replaced by it.
    """
    venv_bin = repo / ".venv" / "bin"
    if not venv_bin.exists():
        raise RuntimeError("no .venv/bin to relocate")

    # 1. Materialize the interpreter inside the venv.
    #
    # ``bin/`` holds a symlink chain (e.g. python -> python3.13 -> external
    # uv interpreter). Resolve EVERY target up front, before mutating anything:
    # once one link is replaced by a real binary copy, ``realpath`` on a sibling
    # that points through it stops inside the venv instead of escaping to the
    # real interpreter prefix — which previously corrupted ``interp_target`` and
    # broke the libpython search below.
    interp_target: Path | None = None
    venv_resolved = (repo / ".venv").resolve()
    resolved = {
        py: Path(os.path.realpath(py))
        for py in venv_bin.glob("python*")
        if py.is_symlink()
    }
    # The true external interpreter is the one whose realpath escapes the venv.
    for py, target in resolved.items():
        try:
            escapes = not target.resolve().is_relative_to(venv_resolved)
        except (OSError, ValueError):
            escapes = False
        if escapes:
            interp_target = target
    for py, target in resolved.items():
        py.unlink()
        shutil.copy2(target, py)
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        print(f"[build_exo] materialized interpreter {py.name} -> {target}")

    # 1b. Bundle libpython next to the interpreter.
    #
    # uv ships a *shared* CPython: bin/python is a small launcher that loads
    # libpythonX.Y.dylib via ``@rpath/libpythonX.Y.dylib`` with an
    # ``LC_RPATH`` of ``@executable_path/../lib``. Copying only the launcher
    # (step 1) leaves that dylib behind, so on the build host it silently
    # resolves to the original uv install while a clean machine dies with
    # ``Library not loaded: @rpath/libpython3.13.dylib``. Copy the dylib into
    # ``.venv/lib/`` so the existing relative rpath resolves anywhere.
    if interp_target is not None:
        src_prefix = interp_target.parent.parent          # .../bin/pythonX.Y -> prefix
        venv_lib = repo / ".venv" / "lib"
        venv_lib.mkdir(parents=True, exist_ok=True)
        dylibs = list((src_prefix / "lib").glob("libpython3.*.dylib"))
        if not dylibs:
            raise RuntimeError(
                f"no libpython3.*.dylib found in {src_prefix / 'lib'} — the "
                "interpreter links @rpath/libpython but the shared library "
                "can't be located to bundle it (the artifact would not run "
                "on a clean machine)."
            )
        for dylib in dylibs:
            dst = venv_lib / dylib.name
            if not dst.exists():
                shutil.copy2(dylib, dst)
                dst.chmod(dst.stat().st_mode | stat.S_IWUSR)
                print(f"[build_exo] bundled {dylib.name} -> {dst.relative_to(repo)}")

        # 1c. Bundle the standard library.
        #
        # A venv contains only ``site-packages`` — its interpreter finds the
        # actual stdlib via ``pyvenv.cfg``'s ``home`` pointing at the base
        # (uv-managed) Python. That path doesn't exist on a user's machine, so
        # the bare venv dies with ``ModuleNotFoundError: No module named
        # 'encodings'``. Copy the base stdlib into ``.venv/lib/pythonX.Y/`` so
        # it travels with the artifact, skipping the base's own site-packages
        # (we keep the venv's, which holds exo + MLX) and dev/GUI/test trees the
        # inference runtime never imports.
        stdlib_dirs = [p for p in (src_prefix / "lib").glob("python3.*") if p.is_dir()]
        if not stdlib_dirs:
            raise RuntimeError(f"no python3.* stdlib found in {src_prefix / 'lib'}")
        src_stdlib = stdlib_dirs[0]
        dst_stdlib = venv_lib / src_stdlib.name
        ignore = shutil.ignore_patterns(
            "site-packages", "__pycache__", "test", "tests",
            "idlelib", "turtledemo", "tkinter", "lib2to3", "ensurepip",
        )
        shutil.copytree(src_stdlib, dst_stdlib, dirs_exist_ok=True, ignore=ignore)
        print(f"[build_exo] bundled stdlib {src_stdlib.name} -> {dst_stdlib.relative_to(repo)}")

        # 1d. Drop pyvenv.cfg so the interpreter resolves its prefix RELATIVE to
        # the executable (``.venv/bin/python`` -> ``.venv/lib/pythonX.Y``)
        # instead of chasing the absent base install. This makes the tree fully
        # relocatable with no absolute paths and no install-time patching: the
        # bundled stdlib and the venv's site-packages both live under
        # ``.venv/lib/pythonX.Y/`` — exactly the layout a non-venv install uses.
        pyvenv_cfg = repo / ".venv" / "pyvenv.cfg"
        if pyvenv_cfg.exists():
            pyvenv_cfg.unlink()
            print("[build_exo] removed pyvenv.cfg (relative stdlib resolution)")

    # 2. Relative-ize console-script shebangs.
    #
    # A venv script normally hard-codes an absolute interpreter path in its
    # shebang. Replace it with a /bin/sh + python polyglot that resolves the
    # interpreter relative to the script's own directory. The body of each
    # console script is Python, so the prologue must be valid in BOTH shells:
    #   * /bin/sh runs ``true`` then ``exec .../python "$0" "$@"`` (lines 2-3),
    #     replacing itself with python before it ever reaches line 4.
    #   * Python sees lines 2-4 as a single triple-quoted string (a no-op),
    #     then runs the real script body.
    # The older one-liner (``"exec" "$(dirname "$0")/python" ...``) is NOT
    # valid Python — it only "worked" while the interpreter failed earlier.
    polyglot = (
        "#!/bin/sh\n"
        "''''true\n"
        'exec "$(dirname "$0")/python" "$0" "$@"\n'
        "'''\n"
    )
    for script in venv_bin.iterdir():
        if script.name.startswith("python") or script.is_dir():
            continue
        try:
            data = script.read_bytes()
        except OSError:
            continue
        if not data.startswith(b"#!"):
            continue
        first_nl = data.find(b"\n")
        body = data[first_nl + 1:] if first_nl != -1 else b""
        if b"python" not in data[:first_nl]:
            continue
        script.write_bytes(polyglot.encode() + body)
        print(f"[build_exo] relocatable shebang: {script.name}")


def _otool_deps(binary: Path) -> list[str]:
    """Return the dylib load paths declared on ``binary`` (via ``otool -L``).

    ``otool -L`` prints the inspected file's path as a header on the first
    line (dropped here). For a dylib/bundle the *second* line is the file's
    own install name (``LC_ID_DYLIB``) rather than a dependency — that is
    NOT filtered out here, so callers must ignore self-references
    themselves (see ``verify_relocatable``/``_install_name``).
    """
    try:
        out = subprocess.run(
            ["otool", "-L", str(binary)],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return []
    lines = out.splitlines()[1:]
    deps = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # "<path> (compatibility version ..., current version ...)"
        path = s.split(" (", 1)[0].strip()
        if path:
            deps.append(path)
    return deps


def _install_name(binary: Path) -> str | None:
    """Return the binary's own install name (``LC_ID_DYLIB``), if it has one.

    ``otool -D`` prints the file-path header followed by the install name.
    Dylibs and many compiled extension bundles carry one (often a stale
    build-tree or delocate-sentinel path such as ``/DLC/...``); plain
    executables do not. This is the library's *own identity*, not a runtime
    dependency, so ``verify_relocatable`` must not mistake it for a
    build-host link.
    """
    try:
        out = subprocess.run(
            ["otool", "-D", str(binary)],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    lines = [ln.strip() for ln in out.splitlines()[1:] if ln.strip()]
    return lines[0] if lines else None


def _is_relocatable_dep(path: str) -> bool:
    """Whether a dylib load path is safe to ship (present on ANY clean Mac).

    Allowed: relative loader-based paths (``@rpath``/``@loader_path``/
    ``@executable_path``) and the two macOS system library trees that
    exist on every install. Anything else — most notably an absolute
    Homebrew path like ``/opt/homebrew/Cellar/...`` or
    ``/usr/local/Cellar/...`` — only exists on the machine that built the
    artifact and will dyld-fail on every other Mac.
    """
    if path.startswith(("@rpath/", "@loader_path/", "@executable_path/")):
        return True
    return path.startswith(("/usr/lib/", "/System/"))


def verify_relocatable(repo: Path) -> None:
    """Fail loudly if the packed ``.venv`` links against anything build-host-specific.

    This is the safety net for the failure mode ``uv_sync``'s
    ``UV_PYTHON_PREFERENCE=only-managed`` is meant to prevent: a Mach-O
    binary inside the runtime (the interpreter, a Rust/pyo3 extension, an
    MLX ``.so``, ...) linked against a Homebrew/system path such as
    ``/opt/homebrew/Cellar/python@3.13/.../Python.framework/...``. That
    dependency only resolves on the exact build machine; on any other Mac
    the artifact dies at launch with a ``dyld: Library not loaded`` error
    instead of a build-time failure — so we check every Mach-O file here
    and refuse to produce an artifact that would ship that bug.
    """
    venv = repo / ".venv"
    if not venv.exists():
        raise RuntimeError(f"no .venv found under {repo} to verify")

    violations: list[str] = []
    for path in venv.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        is_dylib = path.suffix in (".dylib", ".so")
        is_exec = bool(path.stat().st_mode & 0o111)
        if not (is_dylib or is_exec):
            continue

        # A binary's own identity is not a dependency. ``otool -L`` lists a
        # dylib/bundle's install name (``LC_ID_DYLIB``) alongside its real
        # deps, and some Mach-O files even reference their own absolute
        # build-tree path — none of these affect where the loader finds
        # *other* libraries at runtime, so they must never count as a
        # relocatability violation.
        self_refs = {os.path.realpath(path)}
        install_name = _install_name(path)
        if install_name:
            self_refs.add(install_name)

        for dep in _otool_deps(path):
            if _is_relocatable_dep(dep):
                continue
            # Only ABSOLUTE paths can be build-host-specific. Loader-relative
            # paths are handled above; other relative entries (e.g.
            # protobuf's ``bazel-out/...`` load commands) resolve via the
            # binary's rpaths on any machine and are not host-specific.
            if not dep.startswith("/"):
                continue
            if dep in self_refs or os.path.realpath(dep) in self_refs:
                continue
            violations.append(f"  {path.relative_to(repo)}  ->  {dep}")

    if violations:
        raise RuntimeError(
            "exo runtime is NOT relocatable — found Mach-O binaries linked "
            "against build-host-specific paths (these will dyld-fail with "
            "'Library not loaded' on any other machine):\n"
            + "\n".join(violations)
            + "\n\nThis usually means uv_sync() picked up a Homebrew/system "
              "Python instead of a uv-managed one — check UV_PYTHON_PREFERENCE."
        )
    print(f"[build_exo] verified {venv} is relocatable (no build-host-specific dylib paths)")


def pack(repo: Path, ref: str) -> tuple[Path, str]:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    tar_name = f"exo-runtime-{ref}-{arch_tag()}.tar.gz"
    tar_path = DIST_DIR / tar_name

    if tar_path.exists():
        tar_path.unlink()

    top = f"exo-runtime-{ref}"
    print(f"[build_exo] packing {tar_path} …")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(repo, arcname=top)

    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (tar_path.with_suffix(tar_path.suffix + ".sha256")).write_text(
        f"{sha}  {tar_name}\n"
    )
    print(f"[build_exo] sha256 {sha}")
    return tar_path, sha


def build(
    repo_url: str,
    ref: str,
    *,
    skip_clone: bool,
    no_pack: bool = False,
    pack_only: bool = False,
) -> None:
    if platform.system() != "Darwin":
        print("[build_exo] WARNING: artifact is Apple-Silicon only; "
              "building on a non-macOS host will not be usable by Otto.")

    build_dir = DIST_DIR / "src"

    # ``--pack-only`` packs an already-built (and, in CI, already-signed)
    # tree. CI uses: build --no-pack → sign Mach-O in .venv → build --pack-only.
    if pack_only:
        # Defensive: relativize again before packing. ``--pack-only`` packs a
        # tree built earlier (CI signs the venv's Mach-O between build and
        # pack); re-running keeps an absolute build-host exo.pth from ever
        # shipping. Idempotent — already-relative entries are left untouched.
        relativize_pth(build_dir)
        if platform.system() == "Darwin":
            verify_relocatable(build_dir)
        tar_path, sha = pack(build_dir, ref)
        print(f"\n[build_exo] PACKED\n  artifact: {tar_path}\n  sha256:   {sha}")
        return

    if not skip_clone:
        clone_exo(repo_url, ref, build_dir)
    uv_sync(build_dir)
    build_dashboard(build_dir)
    prune(build_dir)
    make_relocatable(build_dir)
    relativize_pth(build_dir)
    if platform.system() == "Darwin":
        verify_relocatable(build_dir)

    if no_pack:
        print(f"\n[build_exo] TREE READY (unpacked): {build_dir}")
        print("  next (CI): codesign+notarize Mach-O inside .venv, then "
              "re-run with --pack-only")
        return

    tar_path, sha = pack(build_dir, ref)
    print(f"\n[build_exo] DONE\n  artifact: {tar_path}\n  sha256:   {sha}")
    print("  next: codesign + notarize the Mach-O files inside, then publish "
          "as a release asset and add it to exo-runtime-manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prebuilt exo runtime artifact")
    parser.add_argument("--repo-url", default=os.environ.get("EXO_REPO_URL", DEFAULT_EXO_REPO_URL))
    parser.add_argument("--ref", default=os.environ.get("EXO_REF", DEFAULT_EXO_REF))
    parser.add_argument("--skip-clone", action="store_true",
                        help="reuse an existing dist/exo/src checkout")
    parser.add_argument("--no-pack", action="store_true",
                        help="build the tree but skip packing (CI signs first)")
    parser.add_argument("--pack-only", action="store_true",
                        help="pack an existing (signed) dist/exo/src tree")
    args = parser.parse_args()
    build(
        args.repo_url,
        args.ref,
        skip_clone=args.skip_clone,
        no_pack=args.no_pack,
        pack_only=args.pack_only,
    )


if __name__ == "__main__":
    main()
