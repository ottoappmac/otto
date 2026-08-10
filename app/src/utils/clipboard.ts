/**
 * Clipboard helpers.
 *
 * ``navigator.clipboard`` is unavailable when the app is served over plain
 * HTTP from a LAN address (non-secure context), so every write falls back to
 * the legacy ``execCommand("copy")`` path via an off-screen textarea.
 */

function legacyCopy(text: string): boolean {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

/** Write plain text to the clipboard. Resolves to false if every path failed. */
export async function copyText(text: string): Promise<boolean> {
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return legacyCopy(text);
  }
}

/**
 * Write an image to the clipboard as a real image (so it can be pasted into
 * documents and chat apps). Falls back to copying `fallbackText` — usually the
 * file URL — when the browser can't put image data on the clipboard.
 */
export async function copyImage(url: string, fallbackText: string): Promise<boolean> {
  try {
    if (typeof ClipboardItem === "undefined") throw new Error("no ClipboardItem");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    let blob = await res.blob();
    // Safari/Chromium only accept a small set of types (png reliably), so
    // anything else is re-encoded through a canvas first.
    if (blob.type !== "image/png") {
      blob = await toPngBlob(blob);
    }
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    return true;
  } catch {
    return copyText(fallbackText);
  }
}

function toPngBlob(blob: Blob): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("no 2d context"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((png) => {
        URL.revokeObjectURL(objectUrl);
        png ? resolve(png) : reject(new Error("encode failed"));
      }, "image/png");
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("decode failed"));
    };
    img.src = objectUrl;
  });
}
