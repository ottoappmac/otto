/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string;
  readonly VITE_OTTO_API_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
