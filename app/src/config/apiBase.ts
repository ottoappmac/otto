export const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:18081";

export const WS_BASE = API_BASE.replace(/^http/, "ws");
