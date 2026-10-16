import { useCallback, useEffect, useState } from "react";
import { FolderPlus, Trash2 } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { FolderGrant, FolderGrantPreview } from "../../types";
import { pickDirectory } from "../../utils/pickFolder";
import { FolderGrantDialog } from "./FolderGrantDialog";

function apiMessage(error: unknown): string {
  const text = error instanceof Error ? error.message : "Could not update folders";
  const json = text.indexOf("{");
  if (json >= 0) {
    try {
      const body = JSON.parse(text.slice(json)) as { error?: string };
      if (body.error) return body.error;
    } catch {
      /* use the raw text */
    }
  }
  return text;
}

export function AllowedFolders() {
  const [grants, setGrants] = useState<FolderGrant[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<FolderGrantPreview | null>(null);
  const [mode, setMode] = useState<"read" | "write">("write");
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api.listFolderGrants().then(setGrants).catch((e) => setError(apiMessage(e)));
  }, []);

  useEffect(() => { load(); }, [load]);

  const startAdd = async () => {
    setError(null);
    const picked = await pickDirectory();
    const path = picked || window.prompt("Folder path to allow");
    if (!path) return;
    try {
      const next = await api.previewFolderGrant(path);
      setMode((next.existing_mode === "read" || next.existing_mode === "write") ? next.existing_mode : "write");
      setPreview(next);
    } catch (e) {
      setError(apiMessage(e));
    }
  };

  const confirm = async () => {
    if (!preview) return;
    setBusy(true);
    try {
      await api.createFolderGrant(preview.realpath, mode);
      setPreview(null);
      load();
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-th-border bg-th-card-bg p-4 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-th-text-primary">Allowed folders</h3>
          <p className="text-[11px] text-th-text-tertiary mt-1 max-w-xl">
            Otto only reads and edits a project folder after you allow it. Shell commands can still read most of the disk. They cannot read SSH keys, keychains, or Otto&apos;s own data, and a read-only folder is not writable from the shell.
          </p>
        </div>
        <button
          type="button"
          onClick={() => { void startAdd(); }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-th-border text-th-text-secondary hover:text-th-text-primary"
        >
          <FolderPlus size={13} />
          Add folder
        </button>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {grants.length === 0 ? (
        <p className="text-xs text-th-text-muted">No folders allowed yet.</p>
      ) : (
        <ul className="space-y-2">
          {grants.map((grant) => (
            <li key={grant.id} className="flex items-center gap-2 text-xs">
              <span className="truncate flex-1 font-mono text-th-text-secondary" title={grant.realpath}>
                {grant.label}
                <span className="text-th-text-muted"> · {grant.realpath}</span>
              </span>
              <button
                type="button"
                onClick={() => {
                  const next = grant.mode === "write" ? "read" : "write";
                  api.updateFolderGrant(grant.id, next).then(load).catch((e) => setError(apiMessage(e)));
                }}
                className="px-2 py-1 rounded border border-th-border text-th-text-tertiary"
              >
                {grant.mode === "read" ? "Read only" : "Can edit"}
              </button>
              <button
                type="button"
                title="Revoke"
                onClick={() => {
                  api.revokeFolderGrant(grant.id).then(load).catch((e) => setError(apiMessage(e)));
                }}
                className="text-th-text-muted hover:text-red-400"
              >
                <Trash2 size={13} />
              </button>
            </li>
          ))}
        </ul>
      )}
      {preview && (
        <FolderGrantDialog
          preview={preview}
          mode={mode}
          busy={busy}
          onMode={setMode}
          onCancel={() => setPreview(null)}
          onConfirm={() => { void confirm(); }}
        />
      )}
    </div>
  );
}
