import { useEffect, useMemo, useState } from "react";
import { api } from "../../hooks/useApi";
import { escapeHtml, highlightCode } from "../../utils/highlightCode";
import { Button } from "../ui/Button";

const HIGHLIGHT_LIMIT = 200_000;

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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (dirty && status !== "saving") void save();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const highlighted = useMemo(
    () => (text.length > HIGHLIGHT_LIMIT ? escapeHtml(text) : highlightCode(text, path)),
    [text, path],
  );
  const lineCount = useMemo(() => text.split("\n").length, [text]);

  if (status === "loading") {
    return <p className="text-sm text-th-text-muted">Loading…</p>;
  }

  const name = path.split("/").pop() ?? path;

  return (
    <div className="h-full min-h-0 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-2xs">
        <span className="flex min-w-0 flex-1 items-center gap-1.5 text-th-text-secondary" title={path}>
          <span className="truncate font-medium text-th-text-primary">{name}</span>
          {dirty && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-th-warning" role="img" aria-label="Unsaved changes" />}
        </span>
        {message && <span className={status === "error" ? "text-th-danger" : "text-th-text-muted"}>{message}</span>}
        <Button
          variant={dirty ? "primary" : "secondary"}
          disabled={!dirty || status === "saving"}
          onClick={() => { void save(); }}
          title="Save (⌘S)"
        >
          {status === "saving" ? "Saving…" : "Save"}
        </Button>
      </div>
      <div className="min-h-[12rem] flex-1 overflow-auto rounded-lg border border-th-border bg-th-inset-bg font-mono text-2xs leading-[1.55]">
        <div className="flex min-h-full min-w-max">
          <div aria-hidden className="sticky left-0 z-10 shrink-0 select-none bg-th-inset-bg py-2 pl-2 pr-2 text-right text-th-text-faint">
            {Array.from({ length: lineCount }, (_, i) => <div key={i}>{i + 1}</div>)}
          </div>
          <div className="relative flex-1 py-2 pr-3">
            <pre aria-hidden className="hljs m-0 whitespace-pre font-[inherit]" dangerouslySetInnerHTML={{ __html: highlighted + "\n " }} />
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              spellCheck={false}
              wrap="off"
              aria-label={`Edit ${name}`}
              className="absolute inset-0 h-full w-full resize-none overflow-hidden whitespace-pre bg-transparent py-2 pr-3 font-[inherit] leading-[inherit] text-transparent caret-th-text-primary outline-none selection:bg-th-accent/30"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
