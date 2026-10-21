import { useEffect, useState } from "react";
import { api } from "../../hooks/useApi";

interface FileEditorProps {
  sessionId: string;
  path: string;
  fileUrl: string;
  revision?: number;
}

/** A text editor for one project file. Saving is the user's edit. */
export function FileEditor({ sessionId, path, fileUrl, revision }: FileEditorProps) {
  const [text, setText] = useState("");
  const [saved, setSaved] = useState("");
  const [status, setStatus] = useState<"loading" | "ready" | "saving" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setMessage("");
    fetch(fileUrl, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`Could not read this file (${response.status})`);
        return response.text();
      })
      .then((body) => {
        if (cancelled) return;
        setText(body);
        setSaved(body);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "Could not read this file");
      });
    return () => { cancelled = true; };
  }, [fileUrl, revision]);

  const dirty = text !== saved;

  const save = async () => {
    setStatus("saving");
    setMessage("");
    try {
      await api.saveWorkspaceFile(sessionId, path, text);
      setSaved(text);
      setStatus("ready");
      setMessage("Saved");
    } catch (error: unknown) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Save failed");
    }
  };

  if (status === "loading") {
    return <p className="text-sm text-th-text-muted">Loading…</p>;
  }

  return (
    <div className="h-full min-h-0 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-[11px]">
        <span className="truncate text-th-text-secondary flex-1" title={path}>{path}</span>
        <button
          type="button"
          disabled={!dirty || status === "saving"}
          onClick={() => { void save(); }}
          className="px-2 py-1 rounded border border-th-border text-th-text-primary disabled:opacity-40"
        >
          {status === "saving" ? "Saving…" : "Save"}
        </button>
      </div>
      {message && <p className="text-[11px] text-th-text-muted">{message}</p>}
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        spellCheck={false}
        className="flex-1 min-h-0 w-full resize-none rounded-lg border border-th-border bg-th-inset-bg p-2 font-mono text-[11px] text-th-text-primary"
      />
    </div>
  );
}
