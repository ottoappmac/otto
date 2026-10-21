import { useEffect, useState } from "react";
import { api } from "../../hooks/useApi";
import type { ActionMode, ActionPolicySettings } from "../../types";

interface MetaCategory {
  id: string;
  title: string;
  blurb: string;
  session_grantable: boolean;
}

interface AuditEntry {
  ts?: string;
  tool?: string;
  category?: string;
  effect?: string;
  reason?: string;
  agent?: string;
}

const PRESET_LABELS: Record<string, string> = {
  locked_down: "Locked down",
  balanced: "Balanced",
  trusting: "Trusting",
};

const EMPTY_POLICY: ActionPolicySettings = {
  preset: "balanced",
  macos_automation_enabled: true,
  categories: {},
  agent_overrides: {},
  rate_limit_per_session: { email_send: 5, message_send: 10, destructive: 10 },
  execute_sandbox: {
    enabled: true,
    network: false,
    extra_read_paths: [],
    extra_write_paths: [],
    env_passthrough: [],
  },
};

export function ActionPolicyPanel({
  policy,
  onChange,
}: {
  policy: ActionPolicySettings | undefined;
  onChange: (next: ActionPolicySettings) => void;
}) {
  const current = policy ?? EMPTY_POLICY;
  const [meta, setMeta] = useState<MetaCategory[]>([]);
  const [presets, setPresets] = useState<Record<string, Record<string, ActionMode>>>({});
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [showAudit, setShowAudit] = useState(false);
  const [overrideAgent, setOverrideAgent] = useState("");
  const [overrideCategory, setOverrideCategory] = useState("shell");
  const [overrideMode, setOverrideMode] = useState<ActionMode>("ask");

  useEffect(() => {
    api.getActionPolicyMeta?.()
      .then((data) => {
        setMeta(data.categories);
        setPresets(data.presets);
      })
      .catch(() => undefined);
  }, []);

  const patch = (next: Partial<ActionPolicySettings>) => onChange({ ...current, ...next });

  const setCategory = (id: string, mode: ActionMode) => {
    if (mode === "allow") {
      const label = meta.find((row) => row.id === id)?.title ?? id;
      const ok = window.confirm(
        `Allow “${label}” without asking? The agent can do this on its own, including after it has read a web page, an email, or a message.`,
      );
      if (!ok) return;
    }
    patch({
      preset: "custom",
      categories: { ...current.categories, [id]: mode },
    });
  };

  const applyPreset = (name: string) => {
    const modes = presets[name];
    if (!modes) return;
    patch({
      preset: name,
      categories: { ...modes },
      macos_automation_enabled: name === "locked_down" ? false : current.macos_automation_enabled,
    });
  };

  const loadAudit = () => {
    setShowAudit(true);
    api.getActionAudit?.()
      .then((data) => setAudit(data.entries))
      .catch(() => setAudit([]));
  };

  const sandbox = current.execute_sandbox ?? EMPTY_POLICY.execute_sandbox;
  const limits = current.rate_limit_per_session ?? EMPTY_POLICY.rate_limit_per_session;

  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-5">
      <h2 className="text-base font-semibold text-th-text-primary">Action policy</h2>
      <p className="text-sm text-th-text-tertiary leading-relaxed">
        Decides what the agent, and every sub-agent, may do. Ask means a card in the chat.
        Unattended runs refuse anything you have not pre-authorised on that schedule or trigger.
      </p>

      <div className="flex flex-wrap gap-2">
        {Object.keys(PRESET_LABELS).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => applyPreset(name)}
            className={`px-3 py-1.5 text-xs rounded-md border ${
              current.preset === name
                ? "bg-th-tab-active-bg text-th-tab-active-fg border-transparent"
                : "border-th-border text-th-text-secondary hover:bg-th-surface-hover"
            }`}
          >
            {PRESET_LABELS[name]}
          </button>
        ))}
        {current.preset === "custom" && (
          <span className="px-3 py-1.5 text-xs text-th-text-muted">Custom</span>
        )}
      </div>

      <label className="flex items-start gap-3 text-sm text-th-text-secondary">
        <input
          type="checkbox"
          className="mt-0.5"
          checked={current.macos_automation_enabled}
          onChange={(event) => patch({ macos_automation_enabled: event.target.checked })}
        />
        <span>
          <span className="font-medium text-th-text-primary">macOS automation</span>
          <span className="block text-xs text-th-text-muted mt-0.5">
            Checked: the agent may use Mail, Messages, AppleScript, and desktop control. Reading the inbox or a chat still asks first. Unchecked: those tools are blocked on the next call, including in this chat.
          </span>
        </span>
      </label>

      <div className="space-y-2">
        {(meta.length ? meta : Object.keys(current.categories).map((id) => ({
          id, title: id, blurb: "", session_grantable: false,
        }))).map((row) => (
          <div key={row.id} className="flex items-center gap-3">
            <div className="flex-1 min-w-0">
              <div className="text-sm text-th-text-primary">{row.title.replace(/\?$/, "")}</div>
              {row.blurb && <div className="text-2xs text-th-text-muted">{row.blurb}</div>}
            </div>
            <select
              className="text-xs bg-th-input-bg border border-th-input-border rounded-md px-2 py-1 text-th-text-primary"
              value={current.categories[row.id] ?? "ask"}
              onChange={(event) => setCategory(row.id, event.target.value as ActionMode)}
            >
              <option value="allow">Allow</option>
              <option value="ask">Ask</option>
              <option value="deny">Deny</option>
            </select>
          </div>
        ))}
      </div>

      <div className="space-y-2 border-t border-th-border pt-4">
        <div className="text-sm font-medium text-th-text-primary">Shell sandbox</div>
        <label className="flex items-center gap-2 text-sm text-th-text-secondary">
          <input
            type="checkbox"
            checked={sandbox.enabled}
            onChange={(event) => patch({
              execute_sandbox: { ...sandbox, enabled: event.target.checked },
            })}
          />
          Sandbox shell commands (Seatbelt). If this is on and the sandbox cannot start, the command is refused.
        </label>
        <label className="flex items-center gap-2 text-sm text-th-text-secondary">
          <input
            type="checkbox"
            checked={sandbox.network}
            onChange={(event) => patch({
              execute_sandbox: { ...sandbox, network: event.target.checked },
            })}
          />
          Allow network from the shell. Off by default, which breaks pip, npm, and git fetch until you turn it on.
        </label>
        <label className="block text-xs text-th-text-muted">
          Extra write paths (one per line)
          <textarea
            className="mt-1 w-full text-xs font-mono bg-th-input-bg border border-th-input-border rounded-md px-2 py-1 text-th-text-primary"
            rows={2}
            value={(sandbox.extra_write_paths ?? []).join("\n")}
            onChange={(event) => patch({
              execute_sandbox: {
                ...sandbox,
                extra_write_paths: event.target.value.split("\n").map((line) => line.trim()).filter(Boolean),
              },
            })}
          />
        </label>
      </div>

      <div className="space-y-2 border-t border-th-border pt-4">
        <div className="text-sm font-medium text-th-text-primary">Session limits</div>
        <p className="text-2xs text-th-text-muted">How many of these one chat may do. 0 means no cap.</p>
        {(["email_send", "message_send", "destructive"] as const).map((key) => (
          <label key={key} className="flex items-center justify-between text-sm text-th-text-secondary">
            <span>{key.replace(/_/g, " ")}</span>
            <input
              type="number"
              min={0}
              className="w-20 text-xs bg-th-input-bg border border-th-input-border rounded-md px-2 py-1"
              value={limits[key] ?? 0}
              onChange={(event) => patch({
                rate_limit_per_session: { ...limits, [key]: Math.max(0, Number(event.target.value) || 0) },
              })}
            />
          </label>
        ))}
      </div>

      <div className="space-y-2 border-t border-th-border pt-4">
        <div className="text-sm font-medium text-th-text-primary">Stricter rules for one agent</div>
        <p className="text-2xs text-th-text-muted">
          An agent can only be tighter than the policy above. A looser override is ignored.
        </p>
        <div className="flex flex-wrap gap-2 items-center">
          <input
            className="text-xs bg-th-input-bg border border-th-input-border rounded-md px-2 py-1"
            placeholder="agent name"
            value={overrideAgent}
            onChange={(event) => setOverrideAgent(event.target.value)}
          />
          <select
            className="text-xs bg-th-input-bg border border-th-input-border rounded-md px-2 py-1"
            value={overrideCategory}
            onChange={(event) => setOverrideCategory(event.target.value)}
          >
            {Object.keys(current.categories).map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
          <select
            className="text-xs bg-th-input-bg border border-th-input-border rounded-md px-2 py-1"
            value={overrideMode}
            onChange={(event) => setOverrideMode(event.target.value as ActionMode)}
          >
            <option value="ask">Ask</option>
            <option value="deny">Deny</option>
          </select>
          <button
            type="button"
            className="text-xs px-2 py-1 rounded-md border border-th-border"
            onClick={() => {
              const name = overrideAgent.trim();
              if (!name) return;
              const prior = current.agent_overrides[name] ?? {};
              patch({
                agent_overrides: {
                  ...current.agent_overrides,
                  [name]: { ...prior, [overrideCategory]: overrideMode },
                },
              });
              setOverrideAgent("");
            }}
          >
            Add
          </button>
        </div>
        {Object.entries(current.agent_overrides ?? {}).map(([agent, rules]) => (
          <div key={agent} className="text-xs text-th-text-secondary flex items-center gap-2">
            <span className="font-mono">{agent}</span>
            <span>{Object.entries(rules).map(([key, mode]) => `${key}=${mode}`).join(", ")}</span>
            <button
              type="button"
              className="text-th-text-muted hover:text-red-400"
              onClick={() => {
                const next = { ...current.agent_overrides };
                delete next[agent];
                patch({ agent_overrides: next });
              }}
            >
              Remove
            </button>
          </div>
        ))}
      </div>

      <div className="border-t border-th-border pt-4">
        <button type="button" className="text-xs text-th-text-secondary underline" onClick={loadAudit}>
          {showAudit ? "Refresh recent actions" : "Show recent actions"}
        </button>
        {showAudit && (
          <ul className="mt-2 space-y-1 max-h-48 overflow-y-auto">
            {audit.length === 0 && <li className="text-xs text-th-text-muted">No decisions recorded yet.</li>}
            {audit.map((entry, index) => (
              <li key={index} className="text-2xs text-th-text-tertiary font-mono">
                {(entry.ts ?? "").slice(0, 19)} {entry.effect} {entry.category} {entry.tool}
                {entry.reason ? ` — ${entry.reason}` : ""}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
