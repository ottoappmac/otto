export const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:18081";

export const WS_BASE = API_BASE.replace(/^http/, "ws");

let apiToken = (import.meta.env.VITE_OTTO_API_TOKEN ?? "").trim();

export function setApiToken(value: string) {
  apiToken = value.trim();
}

export function getApiToken() {
  return apiToken;
}

/** WebSocket URL with the launch token, if the backend is requiring one. */
export function wsUrl(path: string): string {
  const base = `${WS_BASE}${path}`;
  if (!apiToken) return base;
  const join = base.includes("?") ? "&" : "?";
  return `${base}${join}token=${encodeURIComponent(apiToken)}`;
}

export function installApiAuth() {
  const flag = "__ottoFetchPatched";
  const host = window as unknown as Record<string, unknown>;
  if (host[flag]) return;
  host[flag] = true;
  const orig = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
    if (!apiToken || !(url.startsWith(API_BASE) || url.startsWith("/api/"))) {
      return orig(input, init);
    }
    const headers = new Headers(init?.headers);
    if (!headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${apiToken}`);
    }
    return orig(input, { ...init, headers });
  };
}

/** Production builds read the token from the backend's file via Tauri. */
export async function loadApiToken() {
  if (apiToken) return;
  const tauri = (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  if (!tauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const token = await invoke<string>("get_api_token");
      if (token.trim()) {
        setApiToken(token);
        return;
      }
    } catch {
      // The backend may still be writing the file.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}
