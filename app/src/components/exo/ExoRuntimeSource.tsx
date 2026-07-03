// ---------------------------------------------------------------------------
// ExoRuntimeSource — shared "how do we get the Cluster runtime" selector.
//
// One source of truth for the prebuilt-vs-source choice + the custom prebuilt
// runtime URL, reused by:
//   · Settings → LLM → Cluster → Advanced  (with the source repo fields)
//   · First-run onboarding (ClusterSetupFlow) — before the first start
//
// The backend already honours both knobs: `ExoConfig.mode` ("prebuilt" |
// "source") and `ExoConfig.prebuilt_url` flow through `api.exoUp()` /
// `exo_runtime.resolve_artifact()`. This component just surfaces them.
// ---------------------------------------------------------------------------

import type { ExoConfig } from "../../types";

export interface ExoRuntimeSourceProps {
  mode: string;
  prebuiltUrl: string;
  onChange: (patch: Partial<ExoConfig>) => void;
  /**
   * Extra controls rendered inside the "Source & Build" section when the
   * user picks build-from-source (e.g. Settings' repo URL / ref + release
   * check). Onboarding omits these — choosing source there just flips the
   * mode; fine-grained repo tweaks stay in Settings.
   */
  sourceFields?: React.ReactNode;
}

export default function ExoRuntimeSource({
  mode,
  prebuiltUrl,
  onChange,
  sourceFields,
}: ExoRuntimeSourceProps) {
  const isSource = mode === "source";
  return (
    <div className="space-y-5">
      <div className="space-y-1">
        <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
          Runtime
        </p>
        <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onChange({ mode: "prebuilt" })}
              className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                !isSource
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-300"
                  : "border-th-border text-th-text-tertiary"
              }`}
            >
              Prebuilt (recommended)
            </button>
            <button
              type="button"
              onClick={() => onChange({ mode: "source" })}
              className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                isSource
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-300"
                  : "border-th-border text-th-text-tertiary"
              }`}
            >
              Build from source
            </button>
          </div>
          <p className="text-[11px] text-th-text-tertiary leading-relaxed">
            {isSource
              ? "Clones exo and runs uv sync locally. Requires git, uv, node, and a Rust nightly toolchain; first run takes several minutes."
              : "Downloads a notarized, prebuilt runtime on demand (~600 MB, Apple Silicon). No build toolchain needed — Otto fetches it when you start the cluster."}
          </p>
        </div>
      </div>

      {!isSource && (
        <div className="space-y-1">
          <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
            Prebuilt source
          </p>
          <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
            <div>
              <label className="block text-sm font-medium text-th-text-tertiary mb-2">
                Runtime URL
              </label>
              <input
                className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
                type="text"
                value={prebuiltUrl}
                onChange={(e) => onChange({ prebuilt_url: e.target.value })}
                placeholder="https://github.com/ottoappmac/otto/releases/latest/download/exo-runtime-manifest.json"
              />
            </div>
            <p className="text-[11px] text-th-text-tertiary leading-relaxed">
              Where to fetch the prebuilt runtime. Leave blank to use the default GitHub releases.
              Accepts a manifest <code className="font-mono">.json</code> URL, a direct{" "}
              <code className="font-mono">.tar.gz</code> URL, or a{" "}
              <code className="font-mono">file://</code> path for local testing.
            </p>
          </div>
        </div>
      )}

      {isSource && sourceFields && (
        <div className="space-y-1">
          <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">
            Source &amp; Build
          </p>
          <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
            {sourceFields}
          </div>
        </div>
      )}
    </div>
  );
}
