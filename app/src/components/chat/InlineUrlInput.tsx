import {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from "react";

/**
 * A chat composer input that behaves like a textarea but renders pasted/typed
 * URLs as inline, non-editable "chips" that sit right where they appear in the
 * sentence. The canonical value is always a plain string (URLs embedded inline
 * as plain text) so every consumer — draft persistence, slash commands, voice,
 * send — keeps operating on a normal string and never has to know about chips.
 *
 * Implementation notes:
 *  - The element is an uncontrolled `contentEditable` div. Normal typing edits
 *    text nodes directly (so the caret never jumps mid-type). We only rebuild
 *    the DOM when the set of *chipped* URL ranges actually changes.
 *  - A URL only becomes a chip once the caret leaves it (e.g. the user types a
 *    space after it, clicks away, or pastes it) — mirroring how you'd expect a
 *    token to "lock in". The URL token currently under the caret stays plain
 *    text so it remains editable.
 *  - Newlines are kept as `\n` text (the div uses `white-space: pre-wrap`); we
 *    never let the browser inject <br>/<div> so serialization stays trivial.
 */

export interface InlineUrlInputHandle {
  focus: () => void;
  el: HTMLDivElement | null;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLDivElement>) => void;
  /** Called with any files present on a paste. Return true if handled. */
  onPasteFiles?: (files: File[]) => boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

const URL_RE = /https?:\/\/[^\s]+/g;

interface UrlToken {
  start: number;
  end: number;
  url: string;
}

function findUrls(text: string): UrlToken[] {
  const out: UrlToken[] = [];
  URL_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = URL_RE.exec(text))) {
    let url = m[0];
    // Don't swallow trailing sentence punctuation into the chip.
    const trimmed = url.replace(/[).,;:!?'"]+$/, "");
    url = trimmed.length > 0 ? trimmed : url;
    out.push({ start: m.index, end: m.index + url.length, url });
  }
  return out;
}

function displayUrl(url: string): string {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, "");
    const path = u.pathname !== "/" ? u.pathname : "";
    return host + path;
  } catch {
    return url;
  }
}

/** Serialize the editor DOM back into the canonical plain-text value. */
function serialize(root: HTMLElement): string {
  let out = "";
  root.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      out += node.nodeValue ?? "";
    } else if (node instanceof HTMLElement) {
      const url = node.dataset.url;
      if (url !== undefined) out += url;
      else if (node.tagName === "BR") out += "\n";
      else out += node.textContent ?? "";
    }
  });
  return out;
}

/** The chip ranges (by plain-text offset) currently rendered in the DOM. */
function currentChipRanges(root: HTMLElement): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  let offset = 0;
  root.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      offset += (node.nodeValue ?? "").length;
    } else if (node instanceof HTMLElement && node.dataset.url !== undefined) {
      const len = node.dataset.url.length;
      ranges.push([offset, offset + len]);
      offset += len;
    }
  });
  return ranges;
}

function rangesEqual(a: Array<[number, number]>, b: Array<[number, number]>): boolean {
  if (a.length !== b.length) return false;
  return a.every((r, i) => r[0] === b[i][0] && r[1] === b[i][1]);
}

/** Plain-text caret offset within the editor (counting chips as their url length). */
function getCaretOffset(root: HTMLElement): number | null {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0);
  if (!root.contains(range.startContainer) && range.startContainer !== root) return null;

  // Caret anchored directly on the root (e.g. just after a chip): sum the
  // plain-text length of every child before the anchor index.
  if (range.startContainer === root) {
    let offset = 0;
    for (let i = 0; i < range.startOffset && i < root.childNodes.length; i++) {
      offset += plainLength(root.childNodes[i]);
    }
    return offset;
  }

  let offset = 0;
  let found = false;
  const walk = (node: Node): void => {
    if (found) return;
    if (node === range.startContainer && node.nodeType === Node.TEXT_NODE) {
      offset += range.startOffset;
      found = true;
      return;
    }
    if (node.nodeType === Node.TEXT_NODE) {
      offset += (node.nodeValue ?? "").length;
      return;
    }
    if (node instanceof HTMLElement && node.dataset.url !== undefined) {
      offset += node.dataset.url.length;
      return;
    }
    // Element whose children we descend into; also handle caret anchored on it.
    if (node === range.startContainer) {
      // Caret addressed by child index — sum lengths of preceding children.
      for (let i = 0; i < range.startOffset && i < node.childNodes.length; i++) {
        offset += plainLength(node.childNodes[i]);
      }
      found = true;
      return;
    }
    node.childNodes.forEach(walk);
  };
  root.childNodes.forEach(walk);
  return found ? offset : null;
}

function plainLength(node: Node): number {
  if (node.nodeType === Node.TEXT_NODE) return (node.nodeValue ?? "").length;
  if (node instanceof HTMLElement && node.dataset.url !== undefined) return node.dataset.url.length;
  if (node instanceof HTMLElement && node.tagName === "BR") return 1;
  return node.textContent?.length ?? 0;
}

/** Place the caret at a plain-text offset within the editor. */
function setCaretOffset(root: HTMLElement, target: number): void {
  const sel = window.getSelection();
  if (!sel) return;
  let remaining = target;
  const range = document.createRange();
  let placed = false;

  const children = Array.from(root.childNodes);
  for (const node of children) {
    if (placed) break;
    if (node.nodeType === Node.TEXT_NODE) {
      const len = (node.nodeValue ?? "").length;
      if (remaining <= len) {
        range.setStart(node, remaining);
        placed = true;
        break;
      }
      remaining -= len;
    } else if (node instanceof HTMLElement && node.dataset.url !== undefined) {
      const len = node.dataset.url.length;
      if (remaining < len) {
        // Can't sit inside a chip — anchor just before it.
        range.setStartBefore(node);
        placed = true;
        break;
      }
      if (remaining === len) {
        range.setStartAfter(node);
        placed = true;
        break;
      }
      remaining -= len;
    }
  }

  if (!placed) {
    // Past the end — put caret at the very end of the editor.
    range.selectNodeContents(root);
    range.collapse(false);
  } else {
    range.collapse(true);
  }
  sel.removeAllRanges();
  sel.addRange(range);
}

const InlineUrlInput = forwardRef<InlineUrlInputHandle, Props>(function InlineUrlInput(
  { value, onChange, onKeyDown, onPasteFiles, disabled, placeholder, className },
  ref,
) {
  const editorRef = useRef<HTMLDivElement>(null);
  const composingRef = useRef(false);
  // Tracks the value we last emitted from inside the editor, so the reconcile
  // effect can tell internal edits (preserve caret) from external `value`
  // changes like voice/draft restore (caret to end, chip everything).
  const lastEmittedRef = useRef<string | null>(null);
  const caretRef = useRef<number | null>(null);
  // When set, the next reconcile chips every URL regardless of caret position
  // (used right after a paste so a pasted URL becomes a chip immediately).
  const forceChipAllRef = useRef(false);

  useImperativeHandle(ref, () => ({
    focus: () => editorRef.current?.focus(),
    el: editorRef.current,
  }));

  const buildDom = useCallback((text: string, chipAll: boolean, caret: number | null) => {
    const root = editorRef.current;
    if (!root) return;
    const tokens = findUrls(text);
    // Decide which URL tokens render as chips. The token under the caret stays
    // editable text unless we're forcing all (external set / fresh paste).
    const chipTokens = tokens.filter((t) => {
      if (chipAll || caret == null) return true;
      return !(caret >= t.start && caret <= t.end);
    });

    root.replaceChildren();
    let cursor = 0;
    for (const t of chipTokens) {
      if (t.start > cursor) {
        root.appendChild(document.createTextNode(text.slice(cursor, t.start)));
      }
      root.appendChild(makeChip(t.url));
      cursor = t.end;
    }
    if (cursor < text.length) {
      root.appendChild(document.createTextNode(text.slice(cursor)));
    }
    if (root.childNodes.length === 0) {
      // Keep a text node so the caret has somewhere to live.
      root.appendChild(document.createTextNode(""));
    }
  }, []);

  const makeChip = (url: string): HTMLElement => {
    const chip = document.createElement("span");
    chip.dataset.url = url;
    chip.contentEditable = "false";
    chip.title = url;
    chip.className =
      "inline-flex items-center gap-1 align-middle mx-0.5 px-1.5 py-0.5 rounded-md " +
      "bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-medium " +
      "select-none cursor-default max-w-[260px]";

    const label = document.createElement("span");
    label.className = "truncate max-w-[220px]";
    label.textContent = displayUrl(url);
    chip.appendChild(label);

    const close = document.createElement("span");
    close.textContent = "\u00d7";
    close.className = "text-sky-400/60 hover:text-sky-300 leading-none cursor-pointer";
    close.contentEditable = "false";
    close.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      removeUrl(url);
    });
    chip.appendChild(close);
    return chip;
  };

  const removeUrl = (url: string) => {
    const root = editorRef.current;
    if (!root) return;
    const current = serialize(root);
    const idx = current.indexOf(url);
    if (idx === -1) return;
    // Drop the URL plus one adjacent space so we don't leave a double gap.
    let before = current.slice(0, idx);
    let after = current.slice(idx + url.length);
    if (after.startsWith(" ")) after = after.slice(1);
    else if (before.endsWith(" ")) before = before.slice(0, -1);
    const next = before + after;
    caretRef.current = before.length;
    lastEmittedRef.current = next;
    onChange(next);
    requestAnimationFrame(() => editorRef.current?.focus());
  };

  const emitChange = (opts?: { forceChipAll?: boolean; caret?: number | null }) => {
    const root = editorRef.current;
    if (!root) return;
    const text = serialize(root);
    caretRef.current = opts?.caret !== undefined ? opts.caret : getCaretOffset(root);
    if (opts?.forceChipAll) forceChipAllRef.current = true;
    lastEmittedRef.current = text;
    onChange(text);
  };

  // Reconcile the DOM whenever the canonical value changes.
  useLayoutEffect(() => {
    const root = editorRef.current;
    if (!root) return;
    if (composingRef.current) return;

    const isInternal = value === lastEmittedRef.current;
    const chipAll = forceChipAllRef.current || !isInternal;
    const caret = isInternal ? caretRef.current : value.length;

    // Figure out the chip ranges we'd want, then skip the rebuild entirely if
    // the DOM already matches (so normal typing never thrashes the caret).
    const tokens = findUrls(value);
    const desired: Array<[number, number]> = tokens
      .filter((t) => (chipAll || caret == null ? true : !(caret >= t.start && caret <= t.end)))
      .map((t) => [t.start, t.end] as [number, number]);

    const domText = serialize(root);
    if (domText === value && rangesEqual(currentChipRanges(root), desired)) {
      forceChipAllRef.current = false;
      return;
    }

    buildDom(value, chipAll, caret);
    forceChipAllRef.current = false;
    if (document.activeElement === root) {
      setCaretOffset(root, isInternal && caret != null ? caret : value.length);
    }
  }, [value, buildDom]);

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    const files = Array.from(e.clipboardData.items)
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter((f): f is File => f !== null);
    if (files.length > 0 && onPasteFiles?.(files)) {
      e.preventDefault();
      return;
    }
    const text = e.clipboardData.getData("text/plain");
    if (!text) return;
    e.preventDefault();
    insertText(text);
    emitChange({ forceChipAll: true });
  };

  const insertText = (text: string) => {
    const sel = window.getSelection();
    const root = editorRef.current;
    if (!sel || !root || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    range.deleteContents();
    const node = document.createTextNode(text);
    range.insertNode(node);
    range.setStartAfter(node);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    onKeyDown?.(e);
    if (e.defaultPrevented) return;
    if (e.key === "Enter") {
      // Parent didn't claim it (e.g. Shift+Enter newline). Keep the DOM clean
      // by inserting a literal newline instead of letting the browser add <br>.
      e.preventDefault();
      if (e.shiftKey) {
        insertText("\n");
        emitChange();
      }
    }
  };

  return (
    <div className="relative w-full">
      <div
        ref={editorRef}
        role="textbox"
        aria-multiline="true"
        contentEditable={!disabled}
        suppressContentEditableWarning
        spellCheck
        className={className}
        style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}
        onInput={() => emitChange()}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onCompositionStart={() => {
          composingRef.current = true;
        }}
        onCompositionEnd={() => {
          composingRef.current = false;
          emitChange();
        }}
      />
      {value.length === 0 && placeholder && (
        <span className="pointer-events-none absolute left-10 top-3 text-sm text-th-text-muted select-none">
          {placeholder}
        </span>
      )}
    </div>
  );
});

export default InlineUrlInput;
