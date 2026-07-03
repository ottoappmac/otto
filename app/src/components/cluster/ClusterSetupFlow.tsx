// ---------------------------------------------------------------------------
// ClusterSetupFlow — the single, shared "Cluster" (EXO) setup experience.
//
// This component is the one source of truth for bringing a Cluster online,
// reused by:
//   · First-run onboarding (SetupChatPage + SetupWizard) — variant="onboarding"
//   · Settings → LLM → Cluster                            — variant="settings"
//
// Design (Apple HIG):
//   · Progressive disclosure — one decision at a time, logs/advanced hidden.
//   · Sensible defaults — single-Mac by default; recommend one model.
//   · A real finish line — onboarding only reports "Ready" when the runtime
//     is up AND a model is loaded. A quiet "Finish later" clearly flags an
//     incomplete setup instead of scattering "Skip" buttons everywhere.
//   · Consistent language — the feature is always called "Cluster".
//
// The previous three implementations (ExoSetupScreen in SetupWizard, the
// inline handleExoSetup pipeline in SetupChatPage, and the Settings sub-tabs)
// all funnel through this component now.
// ---------------------------------------------------------------------------

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Plus,
  Power,
  RefreshCw,
  Server,
  Sparkles,
  Trash2,
  XCircle,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import ExoModelChooser from "../exo/ExoModelChooser";
import ExoRuntimeSource from "../exo/ExoRuntimeSource";
import ExoOperations from "../ExoOperations";
import type { AppSettings, ExoJob, ExoNodeInfo } from "../../types";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export type ClusterSetupVariant = "onboarding" | "settings";

export interface ClusterSetupFlowProps {
  variant: ClusterSetupVariant;
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  /** Onboarding only — called when the user finishes (or chooses Finish later). */
  onComplete?: () => void;
  /** Onboarding only — called when the user steps back to provider choice. */
  onBack?: () => void;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function cx(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

function toSshAlias(name: string): string {
  return name
    .toLowerCase()
    .replace(/\.local$/, "")
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
}

/** Maps raw backend errors to a friendly title + actionable hint. */
function friendlyClusterError(raw: string | null | undefined): {
  title: string;
  hint: string;
} {
  const s = raw ?? "";
  if (
    /git.*(exit(ed)?|code).*(128|129|130)|fatal.*repository|could not read from remote|connection.*timed out/i.test(
      s,
    )
  ) {
    return {
      title: "Couldn't download the Cluster runtime",
      hint: "This is usually a network, firewall, or VPN issue. Check your connection and try again. Behind a corporate proxy, you may need to allow github.com.",
    };
  }
  if (/uv|pip|python|package/i.test(s) && /(install|build|compile|sync)/i.test(s)) {
    return {
      title: "A dependency failed to install",
      hint: "Try again — if it keeps failing, switch to the prebuilt runtime in Advanced, which needs no build tools.",
    };
  }
  if (/port|address already in use|bind/i.test(s)) {
    return {
      title: "The Cluster port is busy",
      hint: "Another process is using the Cluster API port. Change it under Advanced, or quit the other process and try again.",
    };
  }
  if (s.length > 0) {
    return {
      title: "Cluster setup ran into a problem",
      hint: "Check the details below and try again. You can also finish setup and configure the Cluster later from Settings.",
    };
  }
  return {
    title: "Cluster setup failed",
    hint: "An unknown error occurred. Try again, or finish setup and configure the Cluster later from Settings.",
  };
}

// ---------------------------------------------------------------------------
// FlowStep — a numbered, progressively-disclosed step card (Apple-style).
// ---------------------------------------------------------------------------

function FlowStep({
  num,
  title,
  status,
  doneDetail,
  children,
}: {
  num: number;
  title: string;
  status: "done" | "active" | "pending";
  doneDetail?: string;
  children?: React.ReactNode;
}) {
  const isDone = status === "done";
  const isActive = status === "active";
  return (
    <div
      className={cx(
        "rounded-2xl border transition-colors overflow-hidden",
        isDone
          ? "border-emerald-500/30 bg-emerald-500/5"
          : isActive
            ? "border-th-tab-active-bg/40 bg-th-surface"
            : "border-th-border bg-th-surface opacity-50",
      )}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        <span
          className={cx(
            "inline-flex items-center justify-center w-6 h-6 rounded-full text-[11px] font-bold shrink-0",
            isDone
              ? "bg-emerald-500 text-white"
              : isActive
                ? "bg-th-tab-active-bg text-th-tab-active-fg"
                : "bg-th-inset-bg text-th-text-tertiary border border-th-border",
          )}
        >
          {isDone ? <Check size={12} /> : num}
        </span>
        <div className="flex-1 min-w-0">
          <p
            className={cx(
              "text-sm font-semibold",
              isDone
                ? "text-emerald-600 dark:text-emerald-400"
                : isActive
                  ? "text-th-text-primary"
                  : "text-th-text-tertiary",
            )}
          >
            {title}
          </p>
          {isDone && doneDetail && (
            <p className="text-[10px] text-th-text-tertiary truncate mt-0.5">
              {doneDetail}
            </p>
          )}
        </div>
      </div>
      {isActive && children && (
        <div className="border-t border-th-border/60 px-4 py-3">{children}</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// RuntimeErrorPanel — friendly error with retry + collapsible raw log.
// ---------------------------------------------------------------------------

function RuntimeErrorPanel({
  rawError,
  log,
  onRetry,
}: {
  rawError: string | null;
  log: string[];
  onRetry: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const { title, hint } = friendlyClusterError(rawError);
  const hasDetails = log.length > 0 || !!rawError;
  return (
    <div className="space-y-2.5">
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-3 space-y-1.5">
        <div className="flex items-start gap-2">
          <AlertTriangle size={13} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-[11px] font-semibold text-th-text-primary leading-snug">
            {title}
          </p>
        </div>
        <p className="text-[11px] text-th-text-secondary leading-relaxed pl-5">
          {hint}
        </p>
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
        >
          <RefreshCw size={11} /> Try again
        </button>
        {hasDetails && (
          <button
            type="button"
            onClick={() => setShowDetails((v) => !v)}
            className="text-[11px] text-th-text-muted hover:text-th-text-tertiary underline underline-offset-2 transition-colors"
          >
            {showDetails ? "Hide details" : "Show details"}
          </button>
        )}
      </div>
      {showDetails && hasDetails && (
        <pre className="px-3 py-2.5 rounded-lg border border-th-border bg-th-inset-bg text-[10px] text-th-text-tertiary font-mono whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed">
          {[...log.slice(-40), rawError].filter(Boolean).join("\n")}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cluster node discovery + SSH add (shared by both variants).
// ---------------------------------------------------------------------------

interface MacEntry {
  alias: string;
  label: string;
  host: string;
  user: string;
  port: number;
  status:
    | "idle"
    | "probing"
    | "needs_pw"
    | "installing"
    | "provisioning"
    | "done"
    | "error"
    | "skipped";
  error?: string;
}

function MacCard({
  mac,
  onSetup,
  onInstallKey,
  onSkip,
  onPwChange,
  onUsernameChange,
}: {
  mac: MacEntry;
  onSetup: () => void;
  onInstallKey: () => void;
  onSkip: () => void;
  onPwChange: (pw: string) => void;
  onUsernameChange: (u: string) => void;
}) {
  const [username, setUsername] = useState(mac.user);
  const [pw, setPw] = useState("");
  const prevUser = useRef(mac.user);
  if (mac.user !== prevUser.current) {
    prevUser.current = mac.user;
    setUsername(mac.user);
  }
  if (mac.status === "skipped") return null;

  const isDone = mac.status === "done";
  const isError = mac.status === "error";
  const isNeedsPw = mac.status === "needs_pw";
  const isBusy = ["probing", "installing", "provisioning"].includes(mac.status);
  const busyLabel =
    mac.status === "probing"
      ? "Testing connection…"
      : mac.status === "installing"
        ? "Installing secure key…"
        : "Bringing this Mac into the cluster…";

  function submit() {
    if (!username.trim() || !pw) return;
    onInstallKey();
    setPw("");
  }

  return (
    <div
      className={cx(
        "rounded-xl border px-3 py-2.5 space-y-2.5 transition-colors",
        isDone
          ? "border-emerald-500/30 bg-emerald-500/5"
          : isError
            ? "border-rose-500/20 bg-rose-500/5"
            : "border-th-border bg-th-surface",
      )}
    >
      <div className="flex items-center gap-2">
        {isDone ? (
          <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />
        ) : isError ? (
          <XCircle size={12} className="text-rose-400 shrink-0" />
        ) : isBusy ? (
          <Loader2 size={12} className="animate-spin text-th-text-muted shrink-0" />
        ) : (
          <Server size={12} className="text-th-text-muted shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-medium text-th-text-primary truncate">
            {mac.label}
          </p>
          <p className="text-[10px] text-th-text-muted font-mono">
            {mac.host}
            {mac.port !== 22 ? `:${mac.port}` : ""}
          </p>
        </div>
        {isDone && (
          <span className="text-[10px] font-medium text-emerald-500 shrink-0">
            Added
          </span>
        )}
        {!isDone && !isBusy && !isNeedsPw && (
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={onSetup}
              className="px-2.5 py-1 text-[10px] font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90 transition-opacity"
            >
              Add
            </button>
            <button
              type="button"
              onClick={onSkip}
              className="px-2.5 py-1 text-[10px] font-medium rounded-md border border-th-border text-th-text-tertiary hover:text-th-text-secondary transition-colors"
            >
              Not now
            </button>
          </div>
        )}
      </div>

      {isError && mac.error && (
        <p className="text-[10px] text-rose-400 leading-relaxed">{mac.error}</p>
      )}

      {isNeedsPw && (
        <div className="space-y-2">
          {mac.error && (
            <p className="text-[10px] text-amber-500 leading-relaxed">{mac.error}</p>
          )}
          <p className="text-[10px] text-th-text-muted leading-relaxed">
            Enter the login for this Mac to install a secure key. The password
            is used once and never stored or seen by any AI.
          </p>
          <div className="space-y-1">
            <label className="block text-[10px] font-medium text-th-text-tertiary">
              Username on that Mac
            </label>
            <input
              type="text"
              value={username}
              placeholder="username"
              autoComplete="username"
              onChange={(e) => {
                setUsername(e.target.value);
                onUsernameChange(e.target.value);
              }}
              className="w-full px-2.5 py-1.5 text-xs font-mono rounded-md border border-th-input-border bg-th-input-bg text-th-text-primary placeholder-th-text-muted"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-[10px] font-medium text-th-text-tertiary">
              Mac login password
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                value={pw}
                autoFocus
                placeholder="••••••••"
                autoComplete="current-password"
                onChange={(e) => {
                  setPw(e.target.value);
                  onPwChange(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submit();
                }}
                className="flex-1 px-2.5 py-1.5 text-xs font-mono rounded-md border border-th-input-border bg-th-input-bg text-th-text-primary placeholder-th-text-muted"
              />
              <button
                type="button"
                disabled={!username.trim() || !pw}
                onClick={submit}
                className="px-2.5 py-1.5 text-[10px] font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-40 hover:opacity-90 transition-opacity shrink-0"
              >
                Add Mac
              </button>
            </div>
          </div>
        </div>
      )}

      {isBusy && (
        <p className="text-[10px] text-th-text-muted flex items-center gap-1.5">
          <Loader2 size={9} className="animate-spin" />
          {busyLabel}
        </p>
      )}
    </div>
  );
}

/**
 * Discovers cluster peers (auto-joined via libp2p) and nearby Macs that can
 * be added over SSH. Used inside the onboarding Nodes step and the Settings
 * Cluster tab. Calm by default: a single Mac is a perfectly valid cluster.
 */
function ClusterNodesPanel({
  reachable,
  settings,
  onPatch,
  variant,
}: {
  reachable: boolean;
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  variant: ClusterSetupVariant;
}) {
  const [peers, setPeers] = useState<ExoNodeInfo[]>([]);
  const [macs, setMacs] = useState<MacEntry[]>([]);
  const [scanning, setScanning] = useState(false);
  const [scanned, setScanned] = useState(false);
  const [pubkeyPath, setPubkeyPath] = useState("");
  const pwRefs = useRef<Map<string, string>>(new Map());
  const usernameRefs = useRef<Map<string, string>>(new Map());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const scan = useCallback(async () => {
    if (!reachable) return;
    setScanning(true);
    try {
      const status = await api.exoStatus().catch(() => null);
      const masterId = status?.master_node_id;
      let found = (status?.nodes ?? []).filter((n) => n.node_id !== masterId);
      // Give slow-joining libp2p peers a moment to appear.
      for (let i = 0; i < 3 && found.length === 0; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        const s = await api.exoStatus().catch(() => null);
        found = (s?.nodes ?? []).filter((n) => n.node_id !== s?.master_node_id);
      }
      setPeers(found);

      const existing = new Set(
        (settingsRef.current.exo?.remotes ?? []).map((r) => r.ssh_alias),
      );
      const seen = new Set<string>();
      const list: MacEntry[] = [];
      const [sshRes, lanRes] = await Promise.allSettled([
        api.exoDiscoverSshConfig(),
        api.exoDiscoverLan(4),
      ]);
      if (sshRes.status === "fulfilled") {
        for (const h of sshRes.value.hosts ?? []) {
          if (!seen.has(h.hostname) && !existing.has(h.alias)) {
            seen.add(h.hostname);
            list.push({
              alias: h.alias,
              label: h.alias,
              host: h.hostname,
              user: h.user,
              port: h.port ?? 22,
              status: "idle",
            });
          }
        }
      }
      if (lanRes.status === "fulfilled") {
        for (const h of lanRes.value.hosts ?? []) {
          if (seen.has(h.hostname)) continue;
          const tbAddrs = h.thunderbolt_addresses ?? [];
          const name = h.matches_alias || h.name;
          const alias = toSshAlias(name);
          if (existing.has(alias)) continue;
          seen.add(h.hostname);
          const isTb =
            tbAddrs.length > 0 || h.hostname.startsWith("169.254.");
          const preferredHost = h.hostname.startsWith("169.254.")
            ? h.hostname
            : (tbAddrs[0] ?? h.hostname);
          list.push({
            alias,
            label: isTb ? `${name} (Thunderbolt)` : name,
            host: preferredHost,
            user: "",
            port: h.port ?? 22,
            status: "idle",
          });
        }
      }
      setMacs(list);
    } catch {
      /* non-fatal */
    } finally {
      setScanning(false);
      setScanned(true);
    }
  }, [reachable]);

  useEffect(() => {
    if (reachable && !scanned) void scan();
  }, [reachable, scanned, scan]);

  function updateMac(idx: number, upd: Partial<MacEntry>) {
    setMacs((prev) => prev.map((m, i) => (i === idx ? { ...m, ...upd } : m)));
  }

  async function setupMac(idx: number) {
    const m = macs[idx];
    if (!m) return;
    updateMac(idx, { status: "probing", error: undefined });
    let user = m.user;
    if (!user) {
      user = await api
        .exoSetupLocalUser()
        .then((r) => r.user)
        .catch(() => "");
      updateMac(idx, { user });
    }
    try {
      const probe = await api.exoSetupProbe({ host: m.host, user, port: m.port });
      if (!probe.tcp_reachable) {
        updateMac(idx, {
          status: "error",
          error: `Not reachable on port ${m.port}. Turn on Remote Login on that Mac (System Settings → General → Sharing → Remote Login).`,
        });
        return;
      }
      if (probe.key_auth_ok) {
        await provisionMac(idx, user);
        return;
      }
      if (!probe.password_auth_available) {
        updateMac(idx, {
          status: "error",
          error: "Password sign-in is off on that Mac. Add your SSH key manually, then try again.",
        });
        return;
      }
      let kp = pubkeyPath;
      if (!kp) {
        const existing = await api
          .exoSetupListKeypairs()
          .catch(() => ({ keypairs: [] }));
        kp =
          existing.keypairs.length > 0
            ? existing.keypairs[0].public_path
            : (await api.exoSetupCreateKeypair({
                name: "id_ed25519_exo",
                key_type: "ed25519",
              })).public_path;
        setPubkeyPath(kp);
      }
      updateMac(idx, { status: "needs_pw", user });
    } catch {
      updateMac(idx, {
        status: "error",
        error: "Connection test failed — check that the Mac is on the same network.",
      });
    }
  }

  async function installKey(idx: number) {
    const m = macs[idx];
    if (!m) return;
    const pw = pwRefs.current.get(m.alias) ?? "";
    pwRefs.current.set(m.alias, "");
    const user = usernameRefs.current.get(m.alias)?.trim() || m.user;
    usernameRefs.current.delete(m.alias);
    if (!pw || !pubkeyPath) return;
    if (user !== m.user) updateMac(idx, { user });
    updateMac(idx, { status: "installing" });
    try {
      const result = await api.exoSetupInstallPubkey({
        host: m.host,
        user,
        port: m.port,
        password: pw,
        public_key_path: pubkeyPath,
      });
      if (result.ok || result.already_present) {
        const priv = pubkeyPath.endsWith(".pub")
          ? pubkeyPath.slice(0, -4)
          : pubkeyPath;
        await provisionMac(idx, user, priv);
      } else {
        updateMac(idx, {
          status: "needs_pw",
          error: "Wrong password — use your Mac login password, not your Apple ID.",
        });
      }
    } catch {
      updateMac(idx, {
        status: "needs_pw",
        error: "Couldn't install the key — check the password and try again.",
      });
    }
  }

  async function provisionMac(idx: number, user: string, identityFile?: string) {
    const m = macs[idx];
    if (!m) return;
    updateMac(idx, { status: "provisioning", user });
    try {
      await api
        .addExoRemote({ ssh_alias: m.alias, label: m.label, enabled: true })
        .catch(() => {});
      await api
        .exoSetupAppendSshConfig({
          alias: m.alias,
          hostname: m.host,
          user,
          port: m.port,
          ...(identityFile ? { identity_file: identityFile } : {}),
          replace: true,
        })
        .catch(() => {});
      const job = await api.exoRemoteUp(m.alias);
      let j: ExoJob = job;
      while (j.status !== "done" && j.status !== "error") {
        await new Promise((r) => setTimeout(r, 3000));
        j = await api.getExoJob(j.id);
      }
      if (j.status === "done") {
        updateMac(idx, { status: "done" });
        // Refresh persisted remotes so the configured list reflects the add.
        const fresh = await api.getSettings().catch(() => null);
        if (fresh?.exo?.remotes) {
          await onPatch({ exo: { ...settingsRef.current.exo, remotes: fresh.exo.remotes } });
        }
      } else {
        updateMac(idx, {
          status: "error",
          error: j.error ?? "Setup timed out — try again from Settings.",
        });
      }
    } catch {
      updateMac(idx, {
        status: "error",
        error: "Couldn't finish setup — try again from Settings.",
      });
    }
  }

  const remotes = settings.exo?.remotes ?? [];
  const addedCount =
    peers.length + macs.filter((m) => m.status === "done").length;

  return (
    <div className="space-y-3">
      {/* Auto-discovered peers */}
      {peers.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-emerald-500">
            Already in your cluster
          </p>
          {peers.map((p) => (
            <div
              key={p.node_id}
              className="flex items-center gap-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-3 py-2"
            >
              <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[11px] font-medium text-th-text-primary truncate">
                  {p.friendly_name ?? `${p.node_id.slice(0, 14)}…`}
                </p>
                {(p.chip ?? p.memory_total_gb) && (
                  <p className="text-[10px] text-th-text-muted">
                    {[
                      p.chip,
                      p.memory_total_gb != null &&
                        `${p.memory_total_gb.toFixed(0)} GB`,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Nearby Macs that can be added */}
      {macs.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-th-text-secondary">
            Macs found nearby
          </p>
          {macs.map((m, idx) => (
            <MacCard
              key={m.alias}
              mac={m}
              onSetup={() => void setupMac(idx)}
              onInstallKey={() => void installKey(idx)}
              onSkip={() => updateMac(idx, { status: "skipped" })}
              onPwChange={(pw) => pwRefs.current.set(m.alias, pw)}
              onUsernameChange={(u) => usernameRefs.current.set(m.alias, u)}
            />
          ))}
        </div>
      )}

      {/* Calm single-Mac default */}
      {!scanning && peers.length === 0 && macs.filter((m) => m.status !== "skipped").length === 0 && (
        <div className="rounded-xl border border-th-border bg-th-inset-bg px-3 py-2.5">
          <p className="text-[11px] text-th-text-secondary leading-relaxed">
            Using <span className="font-medium text-th-text-primary">this Mac only</span>
            {" "}— a perfectly good cluster. To add more Macs, connect them over
            Thunderbolt or the same network with Remote Login enabled, then rescan.
          </p>
        </div>
      )}

      {/* Configured remotes (settings variant) */}
      {variant === "settings" && remotes.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-th-text-secondary">
            Configured Macs
          </p>
          {remotes.map((r) => (
            <div
              key={r.ssh_alias}
              className="flex items-center gap-2.5 rounded-xl border border-th-border bg-th-surface px-3 py-2"
            >
              <Server size={12} className="text-th-text-muted shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[11px] font-medium text-th-text-primary truncate">
                  {r.label || r.ssh_alias}
                </p>
                <p className="text-[10px] text-th-text-muted font-mono truncate">
                  {r.ssh_alias}
                </p>
              </div>
              <button
                type="button"
                title="Start this Mac"
                className="p-1 rounded-md text-th-text-muted hover:text-emerald-500"
                onClick={() => void api.exoRemoteUp(r.ssh_alias).catch(() => {})}
              >
                <Power size={12} />
              </button>
              <button
                type="button"
                title="Remove this Mac"
                className="p-1 rounded-md text-th-text-muted hover:text-rose-400"
                onClick={async () => {
                  await api.removeExoRemote(r.ssh_alias).catch(() => {});
                  const fresh = await api.getSettings().catch(() => null);
                  if (fresh?.exo?.remotes) {
                    await onPatch({
                      exo: { ...settingsRef.current.exo, remotes: fresh.exo.remotes },
                    });
                  }
                }}
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={() => void scan()}
        disabled={scanning || !reachable}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-50 transition-colors"
      >
        {scanning ? (
          <Loader2 size={11} className="animate-spin" />
        ) : (
          <Plus size={11} />
        )}
        {scanning
          ? "Looking for Macs…"
          : addedCount > 0
            ? "Add another Mac"
            : "Scan for Macs"}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type RuntimePhase = "probing" | "choosing" | "starting" | "ready" | "error";

export default function ClusterSetupFlow({
  variant,
  settings,
  onPatch,
  onComplete,
  onBack,
}: ClusterSetupFlowProps) {
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [phase, setPhase] = useState<RuntimePhase>("probing");
  const [upJob, setUpJob] = useState<ExoJob | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [showLog, setShowLog] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [reachable, setReachable] = useState(false);
  const [installed, setInstalled] = useState<boolean | null>(null);
  const [modelDone, setModelDone] = useState(!!settings.exo?.model_name);

  // Settings variant lets ExoOperations own the runtime; onboarding auto-starts.
  const isOnboarding = variant === "onboarding";

  // ── Up-job polling ────────────────────────────────────────────────────────
  const logCount = useRef(0);
  useEffect(() => {
    if (!upJob || !["pending", "running"].includes(upJob.status)) return;
    const t = setInterval(async () => {
      try {
        const j = await api.getExoJob(upJob.id);
        const fresh = j.log_lines.slice(logCount.current);
        logCount.current = j.log_lines.length;
        const last = [...fresh].reverse().find((l) => l.trim());
        if (last) setLog((prev) => [...prev.slice(-60), last.trim()]);
        setUpJob(j);
        if (j.status === "done") {
          clearInterval(t);
          const s = await api.exoStatus().catch(() => null);
          setReachable(!!s?.reachable);
          setPhase("ready");
        }
        if (j.status === "error") {
          clearInterval(t);
          setRuntimeError(j.error ?? "The Cluster runtime failed to start.");
          setPhase("error");
        }
      } catch {
        /* transient — retry */
      }
    }, 3000);
    return () => clearInterval(t);
  }, [upJob?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const startNode = useCallback(async (force = false) => {
    setShowLog(true);
    try {
      // Re-read info so the prereq gate reflects the runtime source the user
      // just chose (source mode needs git/uv/node/rust; prebuilt needs none).
      const info = await api.exoInfo();
      if (info.mode_effective === "source") {
        const missing = Object.entries(info.prereqs)
          .filter(([k, v]) => k !== "platform" && k !== "rust_nightly" && !v)
          .map(([k]) => k);
        if (missing.length > 0) {
          setRuntimeError(
            `The Cluster needs a few tools that aren't installed: ${missing.join(", ")}. ` +
              "Switch to the prebuilt runtime above (no build tools needed), or install them and try again.",
          );
          setPhase("error");
          return;
        }
      }
      setPhase("starting");
      // exoUp provisions (downloads/repairs the runtime) when needed, then
      // launches the daemon. `force` re-downloads even if a (possibly broken)
      // runtime is already on disk.
      const job = await api.exoUp(force);
      setUpJob(job);
    } catch (e) {
      setRuntimeError(e instanceof Error ? e.message : "Failed to start the Cluster.");
      setPhase("error");
    }
  }, []);

  const init = useCallback(async () => {
    try {
      // Make sure the Cluster is the active provider for onboarding.
      if (isOnboarding) {
        const exo = settingsRef.current.exo;
        if (!exo?.enabled || settingsRef.current.llm.provider !== "exo") {
          await onPatch({
            exo: { ...exo, enabled: true },
            llm: { ...settingsRef.current.llm, provider: "exo" },
          }).catch(() => undefined);
        }
      }
      const [status, info] = await Promise.all([
        api.exoStatus(),
        api.exoInfo().catch(() => null),
      ]);
      setInstalled(info ? !!info.installed : null);
      if (status.reachable) {
        setReachable(true);
        setPhase("ready");
        return;
      }
      // Let the user confirm the runtime source (prebuilt vs source, custom
      // URL) before kicking off the potentially large first-time download.
      setPhase("choosing");
    } catch (e) {
      setRuntimeError(
        e instanceof Error
          ? e.message
          : "Couldn't reach the Cluster service — there may be a backend issue.",
      );
      setPhase("error");
    }
  }, [isOnboarding, onPatch]);

  useEffect(() => {
    if (isOnboarding) void init();
    else {
      // Settings: reflect live reachability; ExoOperations drives start/stop.
      void api
        .exoStatus()
        .then((s) => setReachable(!!s.reachable))
        .catch(() => undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll reachability in settings so the Nodes/Model sections light up when
  // the user starts the cluster from the operations panel.
  useEffect(() => {
    if (isOnboarding) return;
    const t = setInterval(async () => {
      const s = await api.exoStatus().catch(() => null);
      if (s) setReachable(!!s.reachable);
    }, 4000);
    return () => clearInterval(t);
  }, [isOnboarding]);

  const handleModelChosen = useCallback(
    async (id: string) => {
      await onPatch({ exo: { ...settingsRef.current.exo, model_name: id } });
      setModelDone(true);
    },
    [onPatch],
  );

  // -------------------------------------------------------------------------
  // Settings variant — management sections, no wizard footer.
  // -------------------------------------------------------------------------
  if (!isOnboarding) {
    return (
      <div className="space-y-6">
        <section className="space-y-2">
          <h3 className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
            Cluster
          </h3>
          <ExoOperations enabled={settings.exo.enabled} pane="overview" />
        </section>

        <section className="space-y-2">
          <h3 className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
            Macs in this cluster
          </h3>
          <ClusterNodesPanel
            reachable={reachable}
            settings={settings}
            onPatch={onPatch}
            variant="settings"
          />
        </section>

        <section className="space-y-2">
          <h3 className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
            Model
          </h3>
          <ExoModelChooser
            enabled={settings.exo.enabled && reachable}
            selectedModelId={settings.exo.model_name}
            onUseLoaded={handleModelChosen}
            onPreloadComplete={handleModelChosen}
          />
        </section>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Onboarding variant — numbered steps with a single finish line.
  // -------------------------------------------------------------------------
  const runtimeStatus: "done" | "active" | "pending" =
    phase === "ready" ? "done" : "active";
  const nodesStatus: "done" | "active" | "pending" =
    phase !== "ready" ? "pending" : "active";
  const modelStatus: "done" | "active" | "pending" =
    phase !== "ready" ? "pending" : modelDone ? "done" : "active";

  const busy = phase === "probing" || phase === "starting";

  return (
    <div className="space-y-2">
      {/* Step 1 — Runtime */}
      <FlowStep
        num={1}
        title="Start the cluster"
        status={runtimeStatus}
        doneDetail="Running on this Mac"
      >
        {phase === "probing" && (
          <p className="text-[11px] text-th-text-secondary flex items-center gap-1.5">
            <Loader2 size={11} className="animate-spin" /> Checking the cluster…
          </p>
        )}
        {phase === "choosing" && (
          <div className="space-y-3">
            <p className="text-[11px] text-th-text-secondary leading-relaxed">
              {installed === false
                ? "Otto runs the cluster on this Mac. The first start downloads the runtime (~600 MB for the recommended prebuilt) — no build tools required. You can switch to building from source or point at your own runtime first."
                : "Otto runs the cluster on this Mac. Starting will download the runtime on first use if needed — no build tools required. You can switch to building from source or point at your own runtime first."}
            </p>
            <ExoRuntimeSource
              mode={settings.exo?.mode ?? "prebuilt"}
              prebuiltUrl={settings.exo?.prebuilt_url ?? ""}
              onChange={(patch) => {
                void onPatch({ exo: { ...settingsRef.current.exo, ...patch } });
              }}
            />
            <div className="flex items-center gap-3 flex-wrap">
              <button
                type="button"
                onClick={() => void startNode()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md border border-th-tab-active-bg/40 bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90 transition-colors"
              >
                <Power size={11} />{" "}
                {installed === false ? "Download & start" : "Start the cluster"}
              </button>
              {installed && (
                <button
                  type="button"
                  onClick={() => void startNode(true)}
                  className="inline-flex items-center gap-1.5 text-[11px] text-th-text-muted hover:text-th-text-secondary underline underline-offset-2 transition-colors"
                >
                  <RefreshCw size={10} /> Reinstall runtime
                </button>
              )}
            </div>
          </div>
        )}
        {phase === "starting" && (
          <div className="space-y-2.5">
            <p className="text-[11px] text-th-text-secondary leading-relaxed">
              Setting up the cluster runtime and starting it on this Mac. The
              first time can take a few minutes while it downloads.
            </p>
            <div className="rounded-lg border border-th-border bg-th-inset-bg overflow-hidden">
              <button
                type="button"
                onClick={() => setShowLog((v) => !v)}
                className="w-full flex items-center justify-between px-3 py-1.5 text-[11px] text-th-text-tertiary hover:text-th-text-secondary transition-colors"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Loader2 size={10} className="animate-spin text-th-text-muted" />
                  Starting…
                </span>
                <ChevronDown
                  size={10}
                  className={cx("transition-transform", showLog && "rotate-180")}
                />
              </button>
              {showLog && (
                <div className="border-t border-th-border px-3 py-2">
                  <pre className="text-[10px] text-th-text-tertiary font-mono leading-relaxed whitespace-pre-wrap max-h-36 overflow-y-auto">
                    {log.slice(-30).join("\n") || "(waiting for output…)"}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
        {phase === "error" && (
          <RuntimeErrorPanel
            rawError={runtimeError}
            log={log}
            onRetry={() => {
              setRuntimeError(null);
              setUpJob(null);
              setLog([]);
              logCount.current = 0;
              setPhase("probing");
              void init();
            }}
          />
        )}
      </FlowStep>

      {/* Step 2 — Macs (optional) */}
      <FlowStep
        num={2}
        title="Add more Macs"
        status={nodesStatus}
        doneDetail="Optional"
      >
        <ClusterNodesPanel
          reachable={reachable}
          settings={settings}
          onPatch={onPatch}
          variant="onboarding"
        />
      </FlowStep>

      {/* Step 3 — Model */}
      <FlowStep
        num={3}
        title="Choose a model"
        status={modelStatus}
        doneDetail={settings.exo?.model_name ?? ""}
      >
        <div className="space-y-3">
          <p className="text-[11px] text-th-text-secondary leading-relaxed">
            Pick a model to run on your cluster. We recommend one that fits
            comfortably — add more Macs to unlock bigger models.
          </p>
          <ExoModelChooser
            simple
            enabled={reachable}
            selectedModelId={settings.exo?.model_name}
            onPreloadComplete={handleModelChosen}
            onUseLoaded={handleModelChosen}
          />
        </div>
      </FlowStep>

      {/* Footer — one finish line */}
      <div className="flex items-center justify-between pt-3">
        <button
          type="button"
          onClick={onBack}
          className="px-3 py-2 text-xs font-medium text-th-text-tertiary hover:text-th-text-primary inline-flex items-center gap-1.5"
        >
          <ChevronLeft size={14} /> Back
        </button>
        <div className="flex items-center gap-3">
          {!modelDone && (
            <button
              type="button"
              onClick={onComplete}
              className="text-[11px] text-th-text-tertiary hover:text-th-text-secondary underline underline-offset-2 transition-colors"
            >
              Finish later
            </button>
          )}
          <button
            type="button"
            onClick={onComplete}
            disabled={busy || (!modelDone && phase !== "error")}
            className="px-4 py-2 text-sm font-semibold rounded-lg bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90 disabled:opacity-40 inline-flex items-center gap-1.5 transition-opacity"
          >
            {modelDone ? (
              <>
                <Sparkles size={14} /> You're ready
              </>
            ) : (
              <>
                Continue <ChevronRight size={14} />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
