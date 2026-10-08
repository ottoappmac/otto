// Whether the chat side panel opens itself when the agent reads or writes a
// file. Off by default: files stay available from the changed-files strip,
// the file tree, and clicks on tool results.

const STORAGE_KEY = "otto.followFileEdits";

/** Fired in this window when the preference changes. */
export const FOLLOW_FILE_EDITS_EVENT = "otto-follow-file-edits";

export function getFollowFileEdits(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function setFollowFileEdits(follow: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, follow ? "1" : "0");
  } catch {
    // ignore storage failures — still notify this window
  }
  window.dispatchEvent(new CustomEvent(FOLLOW_FILE_EDITS_EVENT, { detail: follow }));
}
