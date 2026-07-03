import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronRight,
  Clipboard,
  ClipboardCheck,
  Cpu,
  KeyRound,
  Loader2,
  Plus,
  RefreshCw,
  Server,
  ShieldCheck,
  Wifi,
  X,
  XCircle,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { usePolling } from "../../hooks/usePolling";
import type {
  ExoTbLinkSnapshot,
  InstallPubkeyResult,
  LanSshHost,
  LocalKeypair,
  SshConfigAppendResult,
  SshConfigHost,
  SshProbeResult,
} from "../../types";

// ---------------------------------------------------------------------------
// Cluster setup wizard — single-component, inline.
//
// Lives inside the existing "Add remote" panel in SettingsPage. Walks
// the user from a brand-new Mac (no SSH key, not in ~/.ssh/config, only
// reachable by IP / Bonjour / Thunderbolt-Bridge) to a usable EXO
// remote that the existing "Provision & start" button can drive.
//
// Password handling: the password lives only in `pwInputRef.current.value`
// + the inline `installPubkey` body for the duration of one fetch. We
// never copy it into React state and we clear the field on success.
// ---------------------------------------------------------------------------

type Step = 1 | 2 | 3 | 4 | 5 | 6;

export interface ExoSetupCompletion {
  ssh_alias: string;
  hostname: string;
  user: string;
  port: number;
  label: string;
  identity_file: string;
}

interface Props {
  /** Existing ssh-config aliases — used to suggest a unique alias name. */
  sshHosts: SshConfigHost[];
  /** Existing LAN scan results — used to seed the host autocomplete. */
  lanHosts: LanSshHost[];
  /** Already-configured remote aliases — must not collide. */
  existingAliases: string[];
  /** Called when the wizard finishes successfully. The parent should
   *  create the ExoRemoteConfig with these values. */
  onComplete: (result: ExoSetupCompletion) => Promise<void> | void;
  /** Called when the user cancels mid-flow. */
  onCancel: () => void;
}

function classNames(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

function slugifyAlias(input: string): string {
  return (input || "")
    .toLowerCase()
    .replace(/\.local$/, "")
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
}

function dedupeAlias(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  for (let i = 2; i < 50; i++) {
    const c = `${base}-${i}`;
    if (!taken.has(c)) return c;
  }
  return `${base}-${Date.now()}`;
}

function StepHeader({
  index,
  title,
  active,
  done,
}: {
  index: number;
  title: string;
  active: boolean;
  done: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={classNames(
          "w-5 h-5 rounded-full border text-[10px] font-semibold flex items-center justify-center transition-colors",
          done && "bg-emerald-500/20 border-emerald-500/60 text-emerald-500",
          !done && active && "bg-blue-500/20 border-blue-500/60 text-blue-500",
          !done && !active && "bg-th-surface border-th-border text-th-text-muted",
        )}
      >
        {done ? <Check size={11} /> : index}
      </div>
      <span
        className={classNames(
          "text-xs font-medium",
          (active || done) ? "text-th-text-primary" : "text-th-text-muted",
        )}
      >
        {title}
      </span>
    </div>
  );
}

function TrafficLight({ ok, label, hint }: { ok: boolean | null; label: string; hint?: string }) {
  const cls = ok === null
    ? "bg-th-surface text-th-text-muted border-th-border"
    : ok
      ? "bg-emerald-500/15 text-emerald-500 border-emerald-500/30"
      : "bg-rose-500/15 text-rose-500 border-rose-500/30";
  const dot = ok === null ? "bg-neutral-400" : ok ? "bg-emerald-500" : "bg-rose-500";
  return (
    <span
      className={classNames(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 border text-[11px]",
        cls,
      )}
      title={hint}
    >
      <span className={classNames("h-1.5 w-1.5 rounded-full", dot)} />
      {label}
    </span>
  );
}

export default function ExoSetupSteps(props: Props) {
  const { sshHosts, lanHosts, existingAliases, onComplete, onCancel } = props;

  const [step, setStep] = useState<Step>(1);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  // ---------- Step 1: target ----------
  const [host, setHost] = useState("");
  const [user, setUser] = useState("");
  const [port, setPort] = useState(22);
  const [tbSnapshot, setTbSnapshot] = useState<ExoTbLinkSnapshot | null>(null);
  const [tbDismissed, setTbDismissed] = useState(false);

  useEffect(() => {
    void api.exoSetupLocalUser()
      .then((r) => { if (!user) setUser(r.user); })
      .catch(() => undefined);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-prefill the host with the most-likely target when the wizard
  // opens.  We strongly prefer the Bonjour name over any IP because
  // mDNSResponder picks an address it can actually reach right now,
  // sidestepping macOS's notorious per-interface link-local routing.
  useEffect(() => {
    if (host) return;
    const lan = lanHosts.find((h) => h.hostname && h.hostname.endsWith(".local"));
    if (lan?.hostname) {
      setHost(lan.hostname);
      return;
    }
    if (lanHosts[0]) {
      const h = lanHosts[0];
      const ipv4 = (h.addresses || []).find((a) => /^\d+\.\d+\.\d+\.\d+$/.test(a));
      if (h.hostname) setHost(h.hostname);
      else if (ipv4) setHost(ipv4);
    }
  }, [lanHosts, host]);

  // Poll TB-link every 5s while we're still on Step 1 — the user might
  // be plugging the cable in right now. Visibility-aware so the backend
  // is left alone when the window is hidden.
  usePolling(
    async () => {
      try {
        setTbSnapshot(await api.exoTbLink());
      } catch {
        setTbSnapshot(null);
      }
    },
    5000,
    step === 1,
  );

  // ---------- Step 2: probe ----------
  const [probe, setProbe] = useState<SshProbeResult | null>(null);

  const doProbe = useCallback(async () => {
    if (!host.trim() || !user.trim()) return;
    setError(null);
    setBusy("probe");
    try {
      const r = await api.exoSetupProbe({ host: host.trim(), user: user.trim(), port });
      setProbe(r);
      // Show the probe result card (step 2) before auto-advancing so the
      // user can see what was found.
      setStep(2);
      // Auto-advance the suggested next step based on probe.
      if (r.key_auth_ok) {
        // Skip key install entirely; jump to alias step.
        setStep(5);
      } else {
        setStep(3);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Probe failed");
    } finally {
      setBusy(null);
    }
  }, [host, user, port]);

  // ---------- Step 3: keypair ----------
  const [keypairs, setKeypairs] = useState<LocalKeypair[]>([]);
  const [chosenKey, setChosenKey] = useState<string>("");  // private path
  const [newKeyName, setNewKeyName] = useState("id_ed25519_exo");

  const refreshKeypairs = useCallback(async () => {
    try {
      const r = await api.exoSetupListKeypairs();
      setKeypairs(r.keypairs);
      if (!chosenKey && r.keypairs.length > 0) {
        // Prefer a key whose name suggests it's for cluster use.
        const exoKey = r.keypairs.find((k) => /exo|cluster|otto/i.test(k.private_path));
        setChosenKey((exoKey || r.keypairs[0]).private_path);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to list local keys");
    }
  }, [chosenKey]);

  useEffect(() => {
    if (step === 3) void refreshKeypairs();
  }, [step, refreshKeypairs]);

  const doCreateKey = useCallback(async () => {
    setError(null);
    setBusy("create-key");
    try {
      const r = await api.exoSetupCreateKeypair({ name: newKeyName });
      await refreshKeypairs();
      setChosenKey(r.private_path);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Key creation failed");
    } finally {
      setBusy(null);
    }
  }, [newKeyName, refreshKeypairs]);

  const chosenKeypair = useMemo(
    () => keypairs.find((k) => k.private_path === chosenKey) || null,
    [keypairs, chosenKey],
  );

  // ---------- Step 4: install pubkey ----------
  const pwInputRef = useRef<HTMLInputElement | null>(null);
  const [installResult, setInstallResult] = useState<InstallPubkeyResult | null>(null);
  const [showManualFallback, setShowManualFallback] = useState(false);
  const [copied, setCopied] = useState(false);

  const doInstallPubkey = useCallback(async () => {
    if (!chosenKeypair) return;
    const pw = pwInputRef.current?.value || "";
    if (!pw) {
      setError("Enter the password for this one-shot bootstrap.");
      return;
    }
    setError(null);
    setBusy("install");
    try {
      const r = await api.exoSetupInstallPubkey({
        host: host.trim(),
        user: user.trim(),
        port,
        password: pw,
        public_key_path: chosenKeypair.public_path,
        private_key_path: chosenKeypair.private_path,
      });
      setInstallResult(r);
      // If the backend swapped to a Bonjour name because the IP was
      // unroutable, adopt that as the canonical host so ssh-config and
      // verify use the address that actually works.
      if (r.host_used && r.host_used !== host.trim()) {
        setHost(r.host_used);
      }
      // Clear the password field as soon as the response arrives, win
      // or lose. We do this even on failure so a retry must re-type.
      if (pwInputRef.current) pwInputRef.current.value = "";
      setStep(5);
    } catch (e) {
      // Wipe the password field on error too — defence in depth.
      if (pwInputRef.current) pwInputRef.current.value = "";
      setError(e instanceof Error ? e.message : "Install failed");
    } finally {
      setBusy(null);
    }
  }, [chosenKeypair, host, user, port]);

  // ---------- Step 5: alias + ssh-config ----------
  const [alias, setAlias] = useState("");
  const [label, setLabel] = useState("");
  const [appendResult, setAppendResult] = useState<SshConfigAppendResult | null>(null);

  useEffect(() => {
    if (step !== 5) return;
    if (alias) return;
    const taken = new Set(existingAliases.map((a) => a.toLowerCase()));
    sshHosts.forEach((h) => taken.add(h.alias.toLowerCase()));
    const seed = slugifyAlias(probe?.hostname_canonical || host) || "node";
    setAlias(dedupeAlias(seed, taken));
    if (!label && probe?.hostname_canonical) setLabel(probe.hostname_canonical);
  }, [step, alias, existingAliases, sshHosts, probe, host, label]);

  const previewBlock = useMemo(() => {
    if (!alias.trim() || !host.trim()) return "";
    const lines = [`Host ${alias.trim()}`, `  HostName ${host.trim()}`];
    if (user.trim()) lines.push(`  User ${user.trim()}`);
    if (port && port !== 22) lines.push(`  Port ${port}`);
    if (chosenKeypair) {
      lines.push(`  IdentityFile ${chosenKeypair.private_path}`);
      lines.push(`  IdentitiesOnly yes`);
    }
    return lines.join("\n") + "\n";
  }, [alias, host, user, port, chosenKeypair]);

  const doAppendSshConfig = useCallback(async () => {
    if (!alias.trim() || !host.trim()) return;
    setError(null);
    setBusy("append");
    try {
      const r = await api.exoSetupAppendSshConfig({
        alias: alias.trim(),
        hostname: host.trim(),
        user: user.trim() || undefined,
        port: port !== 22 ? port : undefined,
        identity_file: chosenKeypair?.private_path,
        replace: false,
      });
      setAppendResult(r);
      setStep(6);
    } catch (e) {
      setError(e instanceof Error ? e.message : "ssh-config append failed");
    } finally {
      setBusy(null);
    }
  }, [alias, host, user, port, chosenKeypair]);

  // ---------- Step 6: verify + finalize ----------
  const [verify, setVerify] = useState<{ ok: boolean; hint: string } | null>(null);

  const doVerify = useCallback(async () => {
    if (!alias.trim()) return;
    setError(null);
    setBusy("verify");
    try {
      const r = await api.exoTestSsh(alias.trim(), 6.0);
      setVerify({ ok: r.ok, hint: r.hint });
    } catch (e) {
      setVerify({ ok: false, hint: e instanceof Error ? e.message : "Verify failed" });
    } finally {
      setBusy(null);
    }
  }, [alias]);

  useEffect(() => {
    if (step !== 6 || verify !== null) return;
    void doVerify();
  }, [step, verify, doVerify]);

  const doFinalize = useCallback(async () => {
    setError(null);
    setBusy("finalize");
    try {
      await onComplete({
        ssh_alias: alias.trim(),
        hostname: host.trim(),
        user: user.trim(),
        port,
        label: label.trim(),
        identity_file: chosenKeypair?.private_path || "",
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Finalize failed");
    } finally {
      setBusy(null);
    }
  }, [onComplete, alias, host, user, port, label, chosenKeypair]);

  // ---------- Layout ----------
  const stepTitles = [
    "Identify the node",
    "Probe reachability",
    "Choose / create local key",
    "Authorize on remote",
    "Persist alias in ~/.ssh/config",
    "Verify & finalize",
  ];

  return (
    <div className="space-y-3 rounded-md border border-blue-500/40 bg-blue-500/5 p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} className="text-blue-500" />
          <span className="text-xs font-medium text-th-text-primary">
            Set up new node from scratch
          </span>
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="text-th-text-muted hover:text-th-text-primary"
          title="Cancel setup"
        >
          <X size={14} />
        </button>
      </div>

      {/* Progress stepper */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 pb-1 border-b border-th-border">
        {stepTitles.map((t, i) => {
          const idx = (i + 1) as Step;
          const done = idx < step;
          const active = idx === step;
          return (
            <button
              key={idx}
              type="button"
              onClick={() => { if (done) setStep(idx); }}
              disabled={!done}
              className="disabled:cursor-default"
            >
              <StepHeader index={idx} title={t} active={active} done={done} />
            </button>
          );
        })}
      </div>

      {error && (
        <div className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-500 flex items-start gap-2">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* ---------- Step 1: target ---------- */}
      {step === 1 && (
        <div className="space-y-3">
          {tbSnapshot?.connected && !tbDismissed && (() => {
            // If mDNS discovered a .local name covering any of the TB
            // peer addresses, surface that as the *primary* suggestion —
            // it's far more reliable than a raw 169.254.x.y on macOS.
            const tbAddrSet = new Set([
              ...(tbSnapshot.peer_candidates || []),
              ...(tbSnapshot.reachable_peer ? [tbSnapshot.reachable_peer] : []),
            ]);
            const matchingBonjour = lanHosts.find((h) =>
              h.hostname?.endsWith(".local")
              && (h.thunderbolt_addresses || []).some((a) => tbAddrSet.has(a)),
            )?.hostname || "";
            return (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2.5 text-[11px] flex items-start gap-2">
                <Cpu size={14} className="text-amber-500 mt-0.5 shrink-0" />
                <div className="flex-1">
                  <div className="font-medium text-amber-500">
                    Thunderbolt Bridge active ({tbSnapshot.interface})
                  </div>
                  {matchingBonjour ? (
                    <div className="text-th-text-secondary mt-0.5">
                      Peer advertises <code className="font-mono">{matchingBonjour}</code>.
                      <button
                        type="button"
                        className="ml-2 underline text-emerald-500 hover:text-emerald-400"
                        onClick={() => setHost(matchingBonjour)}
                      >
                        Use Bonjour name
                      </button>
                    </div>
                  ) : tbSnapshot.reachable_peer ? (
                    <div className="text-th-text-secondary mt-0.5">
                      Peer reachable at <code className="font-mono">{tbSnapshot.reachable_peer}</code>.
                      <button
                        type="button"
                        className="ml-2 underline text-amber-500 hover:text-amber-400"
                        onClick={() => { if (tbSnapshot.reachable_peer) setHost(tbSnapshot.reachable_peer); }}
                      >
                        Use this address
                      </button>
                    </div>
                  ) : tbSnapshot.peer_candidates && tbSnapshot.peer_candidates.length > 0 ? (
                    <div className="text-th-text-secondary mt-0.5">
                      Peer candidates: {tbSnapshot.peer_candidates.map((c) => (
                        <button
                          key={c}
                          type="button"
                          className="font-mono mr-1.5 underline hover:text-amber-400"
                          onClick={() => setHost(c)}
                        >
                          {c}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="text-th-text-muted mt-0.5">
                      No peer detected yet on the bridge subnet ({tbSnapshot.local_subnets?.join(", ")}).
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  className="text-th-text-muted hover:text-th-text-primary"
                  onClick={() => setTbDismissed(true)}
                >
                  <X size={12} />
                </button>
              </div>
            );
          })()}

          {(sshHosts.length > 0 || lanHosts.length > 0) && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-medium text-th-text-secondary uppercase tracking-wide">
                Quick pick
              </p>
              <div className="flex flex-wrap gap-1.5">
                {sshHosts
                  .filter((h) => !existingAliases.includes(h.alias))
                  .slice(0, 6)
                  .map((h) => (
                    <button
                      key={`s-${h.alias}`}
                      type="button"
                      onClick={() => {
                        setHost(h.hostname || h.alias);
                        if (h.user) setUser(h.user);
                        if (h.port) setPort(h.port);
                      }}
                      className="text-[11px] rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover px-2 py-1"
                      title={`from ~/.ssh/config (${h.source_file})`}
                    >
                      <span className="font-mono">{h.alias}</span>
                      {h.hostname && h.hostname !== h.alias && (
                        <span className="text-th-text-muted ml-1">→ {h.hostname}</span>
                      )}
                    </button>
                  ))}
                {lanHosts.slice(0, 6).map((h) => {
                  // Prefer the Bonjour name (foo.local) — it resolves
                  // through mDNSResponder, which picks whichever address
                  // is actually reachable. Fall back to IPs only when no
                  // .local name was advertised.
                  const tb = (h.thunderbolt_addresses || [])[0];
                  const ipv4 = (h.addresses || []).find((a) => /^\d+\.\d+\.\d+\.\d+$/.test(a));
                  const bonjour = h.hostname && h.hostname.endsWith(".local") ? h.hostname : "";
                  const picked = bonjour || tb || ipv4;
                  const tag = bonjour
                    ? { label: "BONJOUR", cls: "bg-emerald-500/20 text-emerald-500" }
                    : tb
                      ? { label: "TB", cls: "bg-amber-500/20 text-amber-500" }
                      : { label: "IP", cls: "bg-th-surface text-th-text-muted" };
                  return (
                    <button
                      key={`l-${h.name}`}
                      type="button"
                      onClick={() => { if (picked) setHost(picked); }}
                      className="text-[11px] rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover px-2 py-1"
                      title={`Other addresses: ${[tb, ipv4].filter(Boolean).join(", ") || "(none)"}`}
                    >
                      <Wifi size={10} className="inline mr-1 -mt-0.5" />
                      <span className="font-mono">{h.name}</span>
                      <span className={classNames("ml-1 px-1 rounded-sm text-[9px] font-semibold uppercase", tag.cls)}>
                        {tag.label}
                      </span>
                      <span className="text-th-text-muted ml-1 font-mono">→ {picked}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div className="col-span-2">
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                Host or IP
              </label>
              <input
                value={host}
                onChange={(e) => setHost(e.target.value)}
                placeholder="prefer foo.local — IP addresses break on multi-interface Macs"
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary placeholder-th-text-muted"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                Username
              </label>
              <input
                value={user}
                onChange={(e) => setUser(e.target.value)}
                placeholder="eugene"
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary placeholder-th-text-muted"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                Port
              </label>
              <input
                type="number"
                value={port}
                onChange={(e) => setPort(parseInt(e.target.value, 10) || 22)}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary"
              />
            </div>
          </div>

          {/* Progress wheel while probing — shown in-place on step 1 */}
          {busy === "probe" && (
            <div className="flex items-center gap-3 rounded-md border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-[12px]">
              <Loader2 size={16} className="animate-spin text-blue-400 shrink-0" />
              <div>
                <p className="font-medium text-th-text-primary">Identifying node…</p>
                <p className="text-th-text-muted mt-0.5">
                  Connecting to <code className="font-mono">{user}@{host}{port !== 22 ? `:${port}` : ""}</code>
                </p>
              </div>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1.5 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void doProbe()}
              disabled={!host.trim() || !user.trim() || busy === "probe"}
              className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 inline-flex items-center gap-1.5"
            >
              {busy === "probe" ? <Loader2 size={12} className="animate-spin" /> : <ChevronRight size={12} />}
              {busy === "probe" ? "Probing…" : "Probe"}
            </button>
          </div>
        </div>
      )}

      {/* ---------- Step 2: probe result ---------- */}
      {step === 2 && (
        <div className="space-y-3">
          <div className="text-xs text-th-text-secondary">
            Probe result for <code className="font-mono">{user}@{host}{port !== 22 ? `:${port}` : ""}</code>
          </div>
          {busy === "probe" && (
            <div className="flex items-center gap-3 rounded-md border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-[12px]">
              <Loader2 size={16} className="animate-spin text-blue-400 shrink-0" />
              <div>
                <p className="font-medium text-th-text-primary">Identifying node…</p>
                <p className="text-th-text-muted mt-0.5">
                  Connecting to <code className="font-mono">{user}@{host}{port !== 22 ? `:${port}` : ""}</code>
                </p>
              </div>
            </div>
          )}
          {probe && (
            <>
              <div className="flex flex-wrap gap-1.5">
                <TrafficLight ok={probe.tcp_reachable} label="TCP reachable" />
                <TrafficLight ok={probe.key_auth_ok} label="Key auth"
                  hint={probe.key_auth_ok ? "" : "Will need to install a key in step 4."} />
                <TrafficLight ok={probe.key_auth_ok ? probe.has_uv : null} label="uv installed" />
                <TrafficLight ok={probe.key_auth_ok ? probe.has_exo : null} label="exo installed" />
                {probe.os_name && (
                  <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 border border-th-border bg-th-surface text-[11px] text-th-text-secondary">
                    {probe.os_name} {probe.arch}
                  </span>
                )}
              </div>
              {probe.hint && (
                <p className="text-[11px] text-th-text-secondary">{probe.hint}</p>
              )}
              <div className="flex justify-between gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                >
                  Back
                </button>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => void doProbe()}
                    disabled={busy === "probe"}
                    className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover inline-flex items-center gap-1.5"
                  >
                    <RefreshCw size={11} /> Re-probe
                  </button>
                  <button
                    type="button"
                    onClick={() => setStep(probe.key_auth_ok ? 5 : 3)}
                    disabled={!probe.tcp_reachable && !probe.key_auth_ok}
                    className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 inline-flex items-center gap-1.5"
                  >
                    {probe.key_auth_ok ? "Skip to alias" : "Authorize key"} <ChevronRight size={12} />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ---------- Step 3: keypair ---------- */}
      {step === 3 && (
        <div className="space-y-3">
          <p className="text-[11px] text-th-text-secondary">
            Pick a local private key whose public counterpart we'll install on
            the remote, or generate a fresh ED25519 key dedicated to cluster
            use.
          </p>

          {keypairs.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-medium text-th-text-secondary">
                Found in ~/.ssh
              </p>
              <div className="rounded-md border border-th-border bg-th-bg divide-y divide-th-border">
                {keypairs.map((k) => (
                  <label
                    key={k.private_path}
                    className={classNames(
                      "flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-th-surface-hover",
                      chosenKey === k.private_path && "bg-blue-500/10",
                    )}
                  >
                    <input
                      type="radio"
                      name="kp"
                      checked={chosenKey === k.private_path}
                      onChange={() => setChosenKey(k.private_path)}
                      className="accent-blue-600"
                    />
                    <KeyRound size={12} className="text-th-text-tertiary" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-mono text-th-text-primary truncate">
                        {k.private_path.replace(/^.*\/\.ssh\//, "~/.ssh/")}
                      </div>
                      <div className="text-[10px] text-th-text-muted truncate">
                        {k.key_type.toUpperCase()} · {k.bits} · {k.fingerprint}
                        {k.comment ? ` · ${k.comment}` : ""}
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="rounded-md border border-th-border bg-th-bg p-3 space-y-2">
            <p className="text-[11px] font-medium text-th-text-secondary">
              Or generate a fresh ED25519 key
            </p>
            <div className="flex items-center gap-2">
              <code className="text-xs text-th-text-muted">~/.ssh/</code>
              <input
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                className="flex-1 px-2 py-1 bg-th-input-bg border border-th-input-border rounded text-xs font-mono text-th-text-primary"
              />
              <button
                type="button"
                onClick={() => void doCreateKey()}
                disabled={!newKeyName.trim() || busy === "create-key"}
                className="px-3 py-1 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover inline-flex items-center gap-1.5"
              >
                {busy === "create-key" ? <Loader2 size={11} className="animate-spin" /> : <Plus size={11} />}
                Generate
              </button>
            </div>
            <p className="text-[10px] text-th-text-muted leading-relaxed">
              Generated with <code className="font-mono">ssh-keygen -t ed25519 -N ""</code>.
              No passphrase — safe on a FileVault-encrypted Mac.
            </p>
          </div>

          <div className="flex justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={() => setStep(2)}
              className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
            >
              Back
            </button>
            <button
              type="button"
              onClick={() => setStep(4)}
              disabled={!chosenKey}
              className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 inline-flex items-center gap-1.5"
            >
              Continue <ChevronRight size={12} />
            </button>
          </div>
        </div>
      )}

      {/* ---------- Step 4: install pubkey (the password step) ---------- */}
      {step === 4 && (
        <div className="space-y-3">
          {chosenKeypair && (
            <div className="rounded-md border border-th-border bg-th-bg p-2.5 text-[11px] text-th-text-secondary flex items-start gap-2">
              <KeyRound size={12} className="text-th-text-tertiary mt-0.5 shrink-0" />
              <div className="min-w-0">
                <div className="font-mono text-th-text-primary truncate">
                  {chosenKeypair.private_path}
                </div>
                <div className="text-[10px] text-th-text-muted">
                  Will install <code className="font-mono">{chosenKeypair.public_path.split("/").pop()}</code> →
                  <code className="font-mono"> ~/.ssh/authorized_keys</code> on the remote.
                </div>
              </div>
            </div>
          )}

          {!showManualFallback ? (
            <>
              <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2.5 text-[11px] text-th-text-secondary">
                <div className="font-medium text-amber-500 mb-1">
                  One-shot password (not stored, not logged, never sent to the LLM)
                </div>
                <p className="leading-relaxed">
                  Enter the password for{" "}
                  <code className="font-mono">{user}@{host}</code> once. We open a single
                  ssh connection, append the public key, and discard the password
                  immediately. From then on, key auth replaces it for all cluster ops.
                </p>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                  Password
                </label>
                <input
                  ref={pwInputRef}
                  type="password"
                  autoComplete="off"
                  name="exo-bootstrap-pw"
                  className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void doInstallPubkey();
                  }}
                />
              </div>

              <div className="flex justify-between gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setStep(3)}
                  className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                >
                  Back
                </button>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setShowManualFallback(true)}
                    className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                  >
                    I'll do it manually
                  </button>
                  <button
                    type="button"
                    onClick={() => void doInstallPubkey()}
                    disabled={!chosenKeypair || busy === "install"}
                    className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 inline-flex items-center gap-1.5"
                  >
                    {busy === "install" ? <Loader2 size={11} className="animate-spin" /> : <ShieldCheck size={11} />}
                    Install key
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-2">
              <p className="text-[11px] text-th-text-secondary">
                Run one of these on the remote, then come back and re-probe:
              </p>
              <div className="rounded-md border border-th-border bg-th-bg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-medium text-th-text-secondary">
                    From this Mac
                  </span>
                  {chosenKeypair && (
                    <button
                      type="button"
                      onClick={() => {
                        const cmd = `ssh-copy-id -i ${chosenKeypair.public_path} ${user}@${host}${port !== 22 ? ` -p ${port}` : ""}`;
                        navigator.clipboard.writeText(cmd);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 1500);
                      }}
                      className="text-[10px] inline-flex items-center gap-1 text-th-text-muted hover:text-th-text-primary"
                    >
                      {copied ? <ClipboardCheck size={10} /> : <Clipboard size={10} />}
                      {copied ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
                <pre className="text-[11px] font-mono text-th-text-primary overflow-x-auto whitespace-pre-wrap">
{chosenKeypair ? `ssh-copy-id -i ${chosenKeypair.public_path} ${user}@${host}${port !== 22 ? ` -p ${port}` : ""}` : ""}
                </pre>
              </div>
              {chosenKeypair && (
                <p className="text-[10px] text-th-text-muted leading-relaxed">
                  Public key path: <code className="font-mono">{chosenKeypair.public_path}</code> · fingerprint{" "}
                  <code className="font-mono">{chosenKeypair.fingerprint}</code>
                </p>
              )}
              <div className="flex justify-between gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setShowManualFallback(false)}
                  className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                >
                  Back to password
                </button>
                <button
                  type="button"
                  onClick={() => { setStep(2); void doProbe(); }}
                  className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 inline-flex items-center gap-1.5"
                >
                  <RefreshCw size={11} /> I did it — re-probe
                </button>
              </div>
            </div>
          )}

          {installResult && (
            <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-500 flex items-start gap-2">
              <CheckCircle2 size={12} className="mt-0.5 shrink-0" />
              <div>
                <div className="font-medium">
                  {installResult.already_present ? "Key already authorized" : "Key installed"}
                </div>
                <div className="text-[10px]">
                  {installResult.key_type.toUpperCase()} · {installResult.bits} · {installResult.fingerprint}
                </div>
                {installResult.host_swapped_from && installResult.host_used && (
                  <div className="text-[10px] mt-1 text-amber-500">
                    Connected via <code className="font-mono">{installResult.host_used}</code>{" "}
                    after <code className="font-mono">{installResult.host_swapped_from}</code> was unroutable.
                    This is what we'll save into ssh-config.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---------- Step 5: alias + ssh-config ---------- */}
      {step === 5 && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                Alias (~/.ssh/config)
              </label>
              <input
                value={alias}
                onChange={(e) => setAlias(e.target.value)}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1">
                Label (optional)
              </label>
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="Studio Mac"
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary placeholder-th-text-muted"
              />
            </div>
          </div>

          <div className="rounded-md border border-th-border bg-th-bg p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium text-th-text-secondary">
                Block to append to ~/.ssh/config
              </span>
              <span className="text-[10px] text-th-text-muted">
                A timestamped backup will be written first.
              </span>
            </div>
            <pre className="text-[11px] font-mono text-th-text-primary whitespace-pre overflow-x-auto">
{previewBlock}
            </pre>
          </div>

          {appendResult && (
            <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-500">
              Wrote {appendResult.config_path}
              {appendResult.backup_path && (
                <> · backup at <code className="font-mono">{appendResult.backup_path}</code></>
              )}
            </div>
          )}

          <div className="flex justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={() => setStep(probe?.key_auth_ok ? 2 : 4)}
              className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
            >
              Back
            </button>
            <button
              type="button"
              onClick={() => void doAppendSshConfig()}
              disabled={!alias.trim() || !host.trim() || busy === "append"}
              className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 inline-flex items-center gap-1.5"
            >
              {busy === "append" ? <Loader2 size={11} className="animate-spin" /> : <ChevronRight size={11} />}
              Append & continue
            </button>
          </div>
        </div>
      )}

      {/* ---------- Step 6: verify + finalize ---------- */}
      {step === 6 && (
        <div className="space-y-3">
          <div className="rounded-md border border-th-border bg-th-bg p-3">
            <div className="text-[11px] text-th-text-secondary mb-1">
              Verifying alias …
            </div>
            <div className="flex items-center gap-2">
              {busy === "verify" || verify === null ? (
                <>
                  <Loader2 size={12} className="animate-spin text-th-text-muted" />
                  <span className="text-xs text-th-text-muted">
                    Running <code className="font-mono">ssh -o BatchMode=yes {alias} echo …</code>
                  </span>
                </>
              ) : verify.ok ? (
                <>
                  <CheckCircle2 size={12} className="text-emerald-500" />
                  <span className="text-xs text-emerald-500">
                    SSH OK — alias <code className="font-mono">{alias}</code> authenticates with key auth.
                  </span>
                </>
              ) : (
                <>
                  <XCircle size={12} className="text-rose-500" />
                  <span className="text-xs text-rose-500">{verify.hint || "SSH check failed."}</span>
                </>
              )}
            </div>
          </div>

          <div className="flex justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={() => setStep(5)}
              className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
            >
              Back
            </button>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => { setVerify(null); void doVerify(); }}
                disabled={busy === "verify"}
                className="px-3 py-1.5 text-xs rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover inline-flex items-center gap-1.5"
              >
                <RefreshCw size={11} /> Re-verify
              </button>
              <button
                type="button"
                onClick={() => void doFinalize()}
                disabled={!verify?.ok || busy === "finalize"}
                className="px-3 py-1.5 text-xs font-semibold rounded-md bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40 inline-flex items-center gap-1.5"
              >
                {busy === "finalize" ? <Loader2 size={11} className="animate-spin" /> : <Server size={11} />}
                Add to cluster
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
