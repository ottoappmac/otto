import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Folder, FolderPlus, X } from "lucide-react";
import { api } from "../../hooks/useApi";
import { inTauri } from "../../utils/pickFolder";
import type { FolderGrant, SessionWorkspace } from "../../types";

interface ProjectFolderMenuProps {
  workspace: SessionWorkspace | null;
  /** Locks the menu, for example while a turn is running. */
  disabled?: boolean;
  busy?: boolean;
  /** Bind a folder you already allowed. No second confirmation. */
  onUseGrant: (grant: FolderGrant) => void | Promise<void>;
  /**
   * Allow a folder you have not allowed yet. Called with a typed path, or with
   * no argument when the native folder dialog should open.
   */
  onAllowFolder: (path?: string) => void | Promise<void>;
  onClear: () => void | Promise<void>;
  /** The folder was removed from Allowed folders. The current chat is unbound when it was using it. */
  onRevoked?: (grant: FolderGrant) => void | Promise<void>;
  /** Dropdown opens above the trigger when the menu sits near the bottom. */
  placement?: "below" | "above";
}

/**
 * Pick the project folder for a session from the Allowed Folders list, or
 * allow a new one. Chat and Code share this so both behave the same.
 */
export function ProjectFolderMenu({
  workspace,
  disabled,
  busy,
  onUseGrant,
  onAllowFolder,
  onClear,
  onRevoked,
  placement = "below",
}: ProjectFolderMenuProps) {
  const [open, setOpen] = useState(false);
  const [grants, setGrants] = useState<FolderGrant[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const native = inTauri();

  useEffect(() => {
    if (!open) return;
    setLoadError(null);
    api.listFolderGrants().then(setGrants).catch(() => setLoadError("Could not load allowed folders."));
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  const removeGrant = async (grant: FolderGrant) => {
    const wasActive = workspace?.grant_id === grant.id;
    setGrants((prev) => prev.filter((item) => item.id !== grant.id));
    setLoadError(null);
    try {
      await api.revokeFolderGrant(grant.id);
      if (wasActive) await onRevoked?.(grant);
    } catch {
      setLoadError("Could not remove that folder.");
      api.listFolderGrants().then(setGrants).catch(() => setLoadError("Could not load allowed folders."));
    }
  };

  const trigger = workspace ? (
    <>
      <Folder size={11} />
      <span className="truncate">{workspace.name}</span>
      {workspace.mode === "read" && <span className="text-[10px] opacity-70">read</span>}
    </>
  ) : (
    <>
      <FolderPlus size={11} />
      {busy ? "Mapping…" : "Project"}
    </>
  );

  return (
    <div ref={ref} className="relative inline-block min-w-0" data-project-menu>
      <button
        type="button"
        disabled={disabled || busy}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={workspace ? workspace.host_path : "Choose a project folder for the coding agent"}
        className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium max-w-[280px] transition-colors disabled:opacity-60 ${
          workspace
            ? "bg-sky-500/15 text-sky-400 border border-sky-500/25 hover:bg-sky-500/25"
            : "text-th-text-tertiary hover:text-th-text-primary border border-transparent hover:border-th-border hover:bg-th-surface-hover"
        }`}
      >
        {trigger}
        <ChevronDown size={10} className="shrink-0" />
      </button>
      {open && (
        <div
          role="menu"
          className={`absolute left-0 z-50 w-72 max-h-80 overflow-y-auto bg-th-card-bg border border-th-border rounded-lg shadow-xl ${
            placement === "above" ? "bottom-full mb-1" : "top-full mt-1"
          }`}
        >
          <p className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wider text-th-text-muted">Allowed folders</p>
          {loadError && <p className="px-3 py-1 text-xs text-red-400">{loadError}</p>}
          {!loadError && grants.length === 0 && (
            <p className="px-3 py-2 text-xs text-th-text-muted">No folders yet. Allow one below.</p>
          )}
          {grants.map((grant) => {
            const active = workspace?.grant_id === grant.id;
            return (
              <div
                key={grant.id}
                className={`flex items-center pr-1 ${
                  active ? "bg-th-tab-active-bg text-th-tab-active-fg" : "text-th-text-secondary"
                }`}
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => { setOpen(false); void onUseGrant(grant); }}
                  title={grant.realpath}
                  className={`flex-1 min-w-0 text-left px-3 py-1.5 text-xs flex items-center gap-2 transition-colors ${
                    active ? "" : "hover:bg-th-surface-hover hover:text-th-text-primary"
                  }`}
                >
                  <Folder size={12} className="shrink-0" />
                  <span className="truncate flex-1">{grant.label}</span>
                  <span className="text-[10px] text-th-text-muted shrink-0">{grant.mode === "read" ? "read only" : "read/write"}</span>
                  {active && <Check size={12} className="text-emerald-400 shrink-0" />}
                </button>
                <button
                  type="button"
                  aria-label={`Remove ${grant.label}`}
                  title="Remove this folder"
                  onClick={() => { void removeGrant(grant); }}
                  className="shrink-0 p-1 text-th-text-muted hover:text-red-400"
                >
                  <X size={12} />
                </button>
              </div>
            );
          })}
          <div className="border-t border-th-border mt-1 py-1">
            {native ? (
              <button
                type="button"
                role="menuitem"
                onClick={() => { setOpen(false); void onAllowFolder(); }}
                className="w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary"
              >
                <FolderPlus size={12} /> Allow a new folder…
              </button>
            ) : (
              <form
                className="px-3 py-1.5 flex items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  const path = typed.trim();
                  if (!path) return;
                  setTyped("");
                  setOpen(false);
                  void onAllowFolder(path);
                }}
              >
                <input
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  placeholder="/path/to/project"
                  aria-label="Folder path to allow"
                  className="flex-1 min-w-0 bg-th-inset-bg border border-th-border rounded-md px-2 py-1 text-xs text-th-text-primary"
                />
                <button type="submit" disabled={!typed.trim()} className="text-xs text-th-text-secondary hover:text-th-text-primary disabled:opacity-40">
                  Allow
                </button>
              </form>
            )}
            {workspace && (
              <button
                type="button"
                role="menuitem"
                onClick={() => { setOpen(false); void onClear(); }}
                className="w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary"
              >
                <X size={12} /> Remove from this chat
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
