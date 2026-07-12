import { useEffect, useState } from "react";
import { LogIn, LogOut, Plug, Plus, RefreshCw, Trash2, TestTube, X, Wrench, Play, Square, Zap, Pencil, Loader2, CheckCircle2, XCircle, Download, Upload, FileJson, Database, KeyRound, Sparkles, Eye, EyeOff } from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import type { MCPAuthStatus, MCPServerStatus } from "../types";

/** Friendly label for each interactive auth provider. */
const AUTH_KIND_LABEL: Record<string, string> = {
  oauth_device: "OAuth (device flow)",
  oauth_authcode: "OAuth (authorization code)",
  browser_capture: "Browser sign-in",
};

interface TestResult { serverId: string; success: boolean; message: string }

// These servers are managed internally and not surfaced on the Tools page.
const HIDDEN_SERVER_IDS = new Set(["claude-eval-hook", "openclaw-eval-hook", "agent-eval-service"]);

// Servers can expose Start/Stop when there's something for the backend
// to spawn or shut down — either an HTTP MCP that we manage with a
// long-running ManagedProcess (auto_start), or any stdio MCP (the
// langchain-mcp-adapters client owns the subprocess; "Start" maps to
// connect, "Stop" maps to disconnect).
function hasLifecycleControls(srv: MCPServerStatus): boolean {
  return srv.auto_start || srv.transport === "stdio";
}

export default function ToolsPage({ embedded }: { embedded?: boolean } = {}) {
  const [servers, setServers] = useState<MCPServerStatus[]>([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editServer, setEditServer] = useState<MCPServerStatus | null>(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showJsonEditor, setShowJsonEditor] = useState(false);
  const [loading, setLoading] = useState(true);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [connectingId, setConnectingId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestResult | null>(null);

  const refresh = async () => { setLoading(true); try { const data = await api.listMCPServers(); setServers(data); } finally { setLoading(false); } };
  usePolling(refresh, 10000);

  const handleConnect = async (id: string) => { setConnectingId(id); try { await api.connectMCPServer(id); await refresh(); } finally { setConnectingId(null); } };
  const handleTest = async (id: string) => {
    setTestingId(id);
    setTestResult(null);
    try {
      const result = await api.testMCPServer(id);
      setTestResult({ serverId: id, success: result.success, message: result.message });
      setTimeout(() => setTestResult((prev) => prev?.serverId === id ? null : prev), 8000);
    } catch {
      setTestResult({ serverId: id, success: false, message: "Request failed" });
      setTimeout(() => setTestResult((prev) => prev?.serverId === id ? null : prev), 8000);
    } finally {
      setTestingId(null);
    }
  };
  const [confirmRemoveId, setConfirmRemoveId] = useState<string | null>(null);
  const [removeError, setRemoveError] = useState<string | null>(null);
  const handleRemove = async (id: string) => {
    setRemoveError(null);
    try {
      await api.removeMCPServer(id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Delete failed";
      setRemoveError(msg.replace(/^API \d+:\s*/, "").replace(/^\{.*"error":\s*"/, "").replace(/"\}$/, ""));
    }
    setConfirmRemoveId(null);
    await refresh();
  };
  const [startingId, setStartingId] = useState<string | null>(null);
  const [stoppingId, setStoppingId] = useState<string | null>(null);
  const [processError, setProcessError] = useState<string | null>(null);
  const [credentialsServer, setCredentialsServer] = useState<MCPServerStatus | null>(null);

  // The /start route returns 400 with a structured ``missing_credentials``
  // OR ``needs_login`` error when an stdio server isn't ready.  Both
  // map to the same UX (open the credentials dialog) — the dialog
  // itself decides whether to render the text-input list (static creds)
  // or the Login button (interactive auth).
  const tryHandleMissingCredsError = (id: string, errorText: string): boolean => {
    try {
      const m = errorText.match(/\{.*\}$/s);
      if (!m) return false;
      const parsed = JSON.parse(m[0]) as {
        error?: string;
        missing?: string[];
      };
      if (parsed.error !== "missing_credentials" && parsed.error !== "needs_login") {
        return false;
      }
      const srv = servers.find((s) => s.id === id);
      if (srv) {
        setCredentialsServer({
          ...srv,
          missing_secrets: parsed.missing ?? srv.missing_secrets,
        });
      }
      return true;
    } catch {
      return false;
    }
  };

  const handleStart = async (id: string) => {
    setStartingId(id);
    setProcessError(null);
    try {
      const res = await api.startMCPProcess(id) as Record<string, unknown>;
      if (res.error) setProcessError(String(res.error));
      await refresh();
    } catch (e) {
      const errorText = e instanceof Error ? e.message : "Failed to start process";
      if (!tryHandleMissingCredsError(id, errorText)) {
        setProcessError(errorText);
      }
    } finally {
      setStartingId(null);
    }
  };
  const handleStop = async (id: string) => {
    setStoppingId(id);
    setProcessError(null);
    try {
      const res = await api.stopMCPProcess(id) as Record<string, unknown>;
      if (res.error) setProcessError(String(res.error));
      await refresh();
    } catch (e) {
      setProcessError(e instanceof Error ? e.message : "Failed to stop process");
    } finally {
      setStoppingId(null);
    }
  };
  const handleToggleTool = async (serverId: string, toolName: string, exclude: boolean) => {
    const srv = servers.find((s) => s.id === serverId);
    if (!srv) return;
    const newExcluded = exclude
      ? [...srv.excluded_tools, toolName]
      : srv.excluded_tools.filter((t) => t !== toolName);
    setServers((prev) =>
      prev.map((s) => s.id === serverId ? { ...s, excluded_tools: newExcluded } : s),
    );
    await api.updateExcludedTools(serverId, newExcluded);
  };

  const handleExport = async () => {
    const data = await api.exportMCPServers();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "mcp-servers.json"; a.click();
    URL.revokeObjectURL(url);
  };

  const actionButtons = (
    <div className="flex items-center gap-2">
      <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={handleExport}><Download size={14} /> Export</button>
      <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={() => setShowImportDialog(true)}><Upload size={14} /> Import</button>
      <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={() => setShowJsonEditor(true)}><FileJson size={14} /> Edit JSON</button>
      <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={refresh}><RefreshCw size={14} /> Refresh</button>
      <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => setShowAddDialog(true)}><Plus size={15} /> Add MCP</button>
    </div>
  );

  const content = (
    <>
      {!embedded && (
        <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
          <h1 className="text-lg font-bold text-th-text-primary">Tools & Connections</h1>
          {actionButtons}
        </header>
      )}
      {embedded && (
        <div className="px-6 pt-4 pb-3 flex items-center justify-end border-b border-th-border shrink-0">
          {actionButtons}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6">
        {removeError && (
          <div className="mb-4 flex items-center justify-between gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-sm text-red-400">{removeError}</span>
            <button className="text-red-400/60 hover:text-red-400 transition-colors" onClick={() => setRemoveError(null)}><X size={16} /></button>
          </div>
        )}
        {processError && (
          <div className="mb-4 flex items-center justify-between gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-sm text-red-400">{processError}</span>
            <button className="text-red-400/60 hover:text-red-400 transition-colors" onClick={() => setProcessError(null)}><X size={16} /></button>
          </div>
        )}
        {loading && servers.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Loader2 size={32} className="text-th-text-muted animate-spin mb-4" />
            <p className="text-sm text-th-text-tertiary">Loading tools...</p>
          </div>
        )}
        {!loading && servers.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <div className="w-20 h-20 rounded-2xl bg-th-card-bg border border-th-card-border flex items-center justify-center mb-6"><Plug size={36} className="text-th-text-muted" /></div>
            <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No MCP servers</h3>
            <p className="text-sm text-th-text-tertiary mb-6 max-w-xs text-center">Add MCP servers to give your agents access to external tools and services.</p>
            <button className="px-5 py-2.5 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => setShowAddDialog(true)}><Plus size={15} /> Add your first MCP server</button>
          </div>
        )}

        <div className="space-y-3">
          {servers.filter((s) => !s.builtin && !HIDDEN_SERVER_IDS.has(s.id)).map((srv) => (
            <div key={srv.id} className="bg-th-card-bg border border-th-card-border rounded-xl p-5 hover:border-th-border-strong transition-all duration-200">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${srv.connected ? "bg-emerald-500/15 border border-emerald-500/20" : "bg-th-inset-bg border border-th-border"}`}>
                    <Plug size={20} className={srv.connected ? "text-emerald-400" : "text-th-text-muted"} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-th-text-primary">{srv.name}</h3>
                      {srv.generated && (
                        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20"><Sparkles size={9} /> Agent-built</span>
                      )}
                    </div>
                    <p className="text-xs text-th-text-tertiary mt-0.5">
                      {hasLifecycleControls(srv) && <span className={`mr-2 ${srv.process_running ? "text-emerald-400" : "text-th-text-muted"}`}>Process: {srv.process_running ? "running" : "stopped"}</span>}
                      {srv.connected ? `Connected — ${srv.tool_count} tools available` : srv.error ? `Error: ${srv.error}` : "Disconnected"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {srv.context_cache_active && (
                    <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20"><Database size={9} /> Context Cache</span>
                  )}
                  {srv.required_secrets.length > 0 ? (
                    srv.missing_secrets.length > 0 ? (
                      <button
                        type="button"
                        onClick={() => setCredentialsServer(srv)}
                        title="This server needs credentials before it can start"
                        className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/25 hover:bg-amber-500/20 transition-colors"
                      >
                        <KeyRound size={11} />
                        {srv.missing_secrets.length} of {srv.required_secrets.length} credential{srv.required_secrets.length === 1 ? "" : "s"} missing
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setCredentialsServer(srv)}
                        title="Manage stored credentials"
                        className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors"
                      >
                        <KeyRound size={11} />
                        {srv.required_secrets.length} credential{srv.required_secrets.length === 1 ? "" : "s"}
                      </button>
                    )
                  ) : (srv.optional_secrets ?? []).length > 0 ? (
                    <button
                      type="button"
                      onClick={() => setCredentialsServer(srv)}
                      title="Optional credentials — the MCP has a default but you can personalise it"
                      className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-sky-500/10 text-sky-400 border border-sky-500/25 hover:bg-sky-500/20 transition-colors"
                    >
                      <KeyRound size={11} />
                      {(srv.optional_secrets ?? []).length} optional credential{(srv.optional_secrets ?? []).length === 1 ? "" : "s"}
                    </button>
                  ) : srv.auth && srv.auth.kind !== "static" ? (
                    <button
                      type="button"
                      onClick={() => setCredentialsServer(srv)}
                      title={srv.auth.has_bundle && !srv.auth.expired ? "Signed in — click to re-login or logout" : "This server needs an interactive login before it can start"}
                      className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium transition-colors ${
                        srv.auth.has_bundle && !srv.auth.expired
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20"
                          : "bg-amber-500/10 text-amber-400 border border-amber-500/25 hover:bg-amber-500/20"
                      }`}
                    >
                      <KeyRound size={11} />
                      {srv.auth.has_bundle && !srv.auth.expired ? "Signed in" : "Sign-in required"}
                    </button>
                  ) : null}
                  {hasLifecycleControls(srv) && (
                    <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium ${srv.process_running ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" : "bg-th-inset-bg text-th-text-tertiary border border-th-border"}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${srv.process_running ? "bg-emerald-400" : "bg-th-text-muted"}`} />
                      {srv.process_running ? "Running" : "Stopped"}
                    </span>
                  )}
                  {srv.connected ? (
                    <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />Connected</span>
                  ) : !hasLifecycleControls(srv) ? (
                    <span className="inline-flex items-center text-xs px-2.5 py-1 rounded-full font-medium bg-th-inset-bg text-th-text-tertiary border border-th-border">Inactive</span>
                  ) : null}
                </div>
              </div>

              <ToolChips srv={srv} onToggle={handleToggleTool} />

              {testResult?.serverId === srv.id && (
                <div className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium ${testResult.success ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border border-red-500/20 text-red-400"}`}>
                  {testResult.success ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                  {testResult.message}
                </div>
              )}

              <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-th-border">
                {hasLifecycleControls(srv) && (
                  srv.process_running
                    ? <button className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50" onClick={() => handleStop(srv.id)} disabled={stoppingId === srv.id}>{stoppingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <Square size={12} />} {stoppingId === srv.id ? "Stopping..." : "Stop"}</button>
                    : <button
                        className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                        onClick={() => handleStart(srv.id)}
                        disabled={startingId === srv.id}
                        title={srv.missing_secrets.length > 0 ? `Set ${srv.missing_secrets.length} credential(s) first — click Credentials` : undefined}
                      >
                        {startingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                        {startingId === srv.id ? "Starting..." : "Start"}
                      </button>
                )}
                {(srv.required_secrets.length > 0 || (srv.optional_secrets ?? []).length > 0 || (srv.auth && srv.auth.kind !== "static")) && (
                  <button
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                      (srv.optional_secrets ?? []).length > 0 && srv.required_secrets.length === 0
                        ? "bg-sky-500/10 border border-sky-500/25 text-sky-400 hover:bg-sky-500/20"
                        : "bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary"
                    }`}
                    onClick={() => setCredentialsServer(srv)}
                  >
                    <KeyRound size={12} /> Credentials
                  </button>
                )}
                <button
                  className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                  onClick={() => handleConnect(srv.id)}
                  disabled={connectingId === srv.id}
                >
                  {connectingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                  {connectingId === srv.id ? "Connecting..." : srv.connected ? "Reconnect" : "Connect"}
                </button>
                <button
                  className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                  onClick={() => handleTest(srv.id)}
                  disabled={testingId === srv.id}
                >
                  {testingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
                  {testingId === srv.id ? "Testing..." : "Test"}
                </button>
                <button className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5" onClick={() => setEditServer(srv)}><Pencil size={12} /> Edit</button>
                {!srv.builtin && (
                  confirmRemoveId === srv.id ? (
                    <span className="flex items-center gap-1.5">
                      <span className="text-xs text-red-400">{srv.generated ? "Remove server, source file, and credentials?" : "Remove?"}</span>
                      <button className="px-2.5 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 rounded-lg text-xs font-semibold transition-colors" onClick={() => handleRemove(srv.id)}>Yes</button>
                      <button className="px-2.5 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors" onClick={() => setConfirmRemoveId(null)}>No</button>
                    </span>
                  ) : (
                    <button className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5" onClick={() => setConfirmRemoveId(srv.id)}><Trash2 size={12} /> Remove</button>
                  )
                )}
              </div>
            </div>
          ))}

          {servers.some((s) => s.builtin && !HIDDEN_SERVER_IDS.has(s.id)) && (
            <>
              <div className="flex items-center gap-3 mt-6 mb-3">
                <div className="flex-1 border-t border-th-border" />
                <span className="text-sm uppercase tracking-widest text-th-text-muted font-semibold flex items-center gap-1.5"><Zap size={14} /> Managed</span>
                <div className="flex-1 border-t border-th-border" />
              </div>
              {servers.filter((s) => s.builtin && !HIDDEN_SERVER_IDS.has(s.id)).map((srv) => (
                <div key={srv.id} className={`bg-th-card-bg border border-th-card-border rounded-xl p-5 transition-all duration-200 ${srv.os_supported ? "hover:border-th-border-strong" : "opacity-40 pointer-events-none"}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${srv.connected ? "bg-emerald-500/15 border border-emerald-500/20" : "bg-th-inset-bg border border-th-border"}`}>
                        <Plug size={20} className={srv.connected ? "text-emerald-400" : "text-th-text-muted"} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-th-text-primary">{srv.name}</h3>
                          {!srv.os_supported && srv.requires_os && (
                            <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded-full font-semibold bg-th-inset-bg text-th-text-tertiary border border-th-border">
                              {srv.requires_os.charAt(0).toUpperCase() + srv.requires_os.slice(1)} only
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-th-text-tertiary mt-0.5">
                          {!srv.os_supported ? "Not available on this platform" : (
                            <>
                              {srv.auto_start && <span className={`mr-2 ${srv.process_running ? "text-emerald-400" : "text-th-text-muted"}`}>Process: {srv.process_running ? "running" : "stopped"}</span>}
                              {srv.connected ? `Connected — ${srv.tool_count} tools available` : srv.error ? `Error: ${srv.error}` : "Disconnected"}
                            </>
                          )}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {srv.os_supported && srv.context_cache_active && (
                        <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20"><Database size={9} /> Context Cache</span>
                      )}
                      {srv.os_supported && (
                        srv.required_secrets.length > 0 ? (
                          srv.missing_secrets.length > 0 ? (
                            <button
                              type="button"
                              onClick={() => setCredentialsServer(srv)}
                              title="This server needs credentials before it can start"
                              className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/25 hover:bg-amber-500/20 transition-colors"
                            >
                              <KeyRound size={11} />
                              {srv.missing_secrets.length} of {srv.required_secrets.length} credential{srv.required_secrets.length === 1 ? "" : "s"} missing
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setCredentialsServer(srv)}
                              title="Manage stored credentials"
                              className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors"
                            >
                              <KeyRound size={11} />
                              {srv.required_secrets.length} credential{srv.required_secrets.length === 1 ? "" : "s"}
                            </button>
                          )
                        ) : (srv.optional_secrets ?? []).length > 0 ? (
                          <button
                            type="button"
                            onClick={() => setCredentialsServer(srv)}
                            title="Optional credentials — the MCP has a default but you can personalise it"
                            className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-sky-500/10 text-sky-400 border border-sky-500/25 hover:bg-sky-500/20 transition-colors"
                          >
                            <KeyRound size={11} />
                            {(srv.optional_secrets ?? []).length} optional credential{(srv.optional_secrets ?? []).length === 1 ? "" : "s"}
                          </button>
                        ) : srv.auth && srv.auth.kind !== "static" ? (
                          <button
                            type="button"
                            onClick={() => setCredentialsServer(srv)}
                            title={srv.auth.has_bundle && !srv.auth.expired ? "Signed in — click to re-login or logout" : "This server needs an interactive login before it can start"}
                            className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium transition-colors ${
                              srv.auth.has_bundle && !srv.auth.expired
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20"
                                : "bg-amber-500/10 text-amber-400 border border-amber-500/25 hover:bg-amber-500/20"
                            }`}
                          >
                            <KeyRound size={11} />
                            {srv.auth.has_bundle && !srv.auth.expired ? "Signed in" : "Sign-in required"}
                          </button>
                        ) : null
                      )}
                      {srv.os_supported && srv.auto_start && (
                        <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium ${srv.process_running ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" : "bg-th-inset-bg text-th-text-tertiary border border-th-border"}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${srv.process_running ? "bg-emerald-400" : "bg-th-text-muted"}`} />
                          {srv.process_running ? "Running" : "Stopped"}
                        </span>
                      )}
                      {srv.os_supported && (srv.connected ? (
                        <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />Connected</span>
                      ) : !srv.auto_start ? (
                        <span className="inline-flex items-center text-xs px-2.5 py-1 rounded-full font-medium bg-th-inset-bg text-th-text-tertiary border border-th-border">Inactive</span>
                      ) : null)}
                    </div>
                  </div>

                  {srv.os_supported && <ToolChips srv={srv} onToggle={handleToggleTool} />}

                  {srv.os_supported && testResult?.serverId === srv.id && (
                    <div className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium ${testResult.success ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border border-red-500/20 text-red-400"}`}>
                      {testResult.success ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                      {testResult.message}
                    </div>
                  )}

                  {srv.os_supported && (
                    <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-th-border">
                      {srv.auto_start && (
                        srv.process_running
                          ? <button className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50" onClick={() => handleStop(srv.id)} disabled={stoppingId === srv.id}>{stoppingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <Square size={12} />} {stoppingId === srv.id ? "Stopping..." : "Stop"}</button>
                          : <button className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50" onClick={() => handleStart(srv.id)} disabled={startingId === srv.id}>{startingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />} {startingId === srv.id ? "Starting..." : "Start"}</button>
                      )}
                      <button
                        className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                        onClick={() => handleConnect(srv.id)}
                        disabled={connectingId === srv.id}
                      >
                        {connectingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                        {connectingId === srv.id ? "Connecting..." : srv.connected ? "Reconnect" : "Connect"}
                      </button>
                      <button
                        className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                        onClick={() => handleTest(srv.id)}
                        disabled={testingId === srv.id}
                      >
                        {testingId === srv.id ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
                        {testingId === srv.id ? "Testing..." : "Test"}
                      </button>
                      {(srv.required_secrets.length > 0 || (srv.optional_secrets ?? []).length > 0 || (srv.auth && srv.auth.kind !== "static")) && (
                        <button
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                            (srv.optional_secrets ?? []).length > 0 && srv.required_secrets.length === 0
                              ? "bg-sky-500/10 border border-sky-500/25 text-sky-400 hover:bg-sky-500/20"
                              : "bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary"
                          }`}
                          onClick={() => setCredentialsServer(srv)}
                        >
                          <KeyRound size={12} /> Credentials
                        </button>
                      )}
                      <button className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5" onClick={() => setEditServer(srv)}><Pencil size={12} /> Edit</button>
                    </div>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </div>
      {showAddDialog && <AddMCPDialog onClose={() => setShowAddDialog(false)} onAdded={() => { setShowAddDialog(false); refresh(); }} />}
      {editServer && <EditMCPDialog server={editServer} onClose={() => setEditServer(null)} onSaved={() => { setEditServer(null); refresh(); }} />}
      {showImportDialog && <ImportMCPDialog onClose={() => setShowImportDialog(false)} onImported={() => { setShowImportDialog(false); refresh(); }} />}
      {showJsonEditor && <EditJsonDialog servers={servers} onClose={() => setShowJsonEditor(false)} onSaved={() => { setShowJsonEditor(false); refresh(); }} />}
      {credentialsServer && (
        <CredentialsDialog
          server={credentialsServer}
          onClose={() => setCredentialsServer(null)}
          onChanged={async () => { await refresh(); }}
        />
      )}
    </>
  );

  if (embedded) return <>{content}</>;
  return <div className="h-full flex flex-col">{content}</div>;
}

function AddMCPDialog({ onClose, onAdded }: { onClose: () => void; onAdded: () => void }) {
  const [name, setName] = useState("");
  const [transport, setTransport] = useState("streamable_http");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [saving, setSaving] = useState(false);

  const handleAdd = async () => { setSaving(true); try { const cleanCommand = command.trim().replace(/^["']+|["']+$/g, ""); await api.addMCPServer({ name, transport, url: transport !== "stdio" ? url : null, command: transport === "stdio" ? cleanCommand || null : null, args: transport === "stdio" ? args.split(/\s+/).filter(Boolean) : [] }); onAdded(); } finally { setSaving(false); } };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-lg p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">Add MCP Server</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>
        <div className="space-y-4">
          <Field label="Name" value={name} onChange={setName} placeholder="My MCP Server" />
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Transport</label>
            <select className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 text-sm" value={transport} onChange={(e) => setTransport(e.target.value)}>
              <option value="streamable_http">Streamable HTTP</option>
              <option value="stdio">stdio (local command)</option>
              <option value="sse">SSE (legacy)</option>
            </select>
          </div>
          {transport !== "stdio" && <Field label="URL" value={url} onChange={setUrl} placeholder="http://localhost:3000/mcp" />}
          {transport === "stdio" && (<><Field label="Command (no quotes needed)" value={command} onChange={setCommand} placeholder="npx" /><Field label="Arguments (space-separated)" value={args} onChange={setArgs} placeholder="@modelcontextprotocol/server-github" /></>)}
        </div>
        <div className="flex justify-end gap-2 mt-6 pt-4 border-t border-th-border">
          <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
          <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40" onClick={handleAdd} disabled={!name || saving}>{saving ? "Adding..." : "Add Server"}</button>
        </div>
      </div>
    </div>
  );
}

function EditMCPDialog({ server, onClose, onSaved }: { server: MCPServerStatus; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(server.name);
  const [transport, setTransport] = useState(server.transport);
  const [url, setUrl] = useState(server.url ?? "");
  const [port, setPort] = useState(server.port != null ? String(server.port) : "");
  const [command, setCommand] = useState(server.command ?? "");
  const [args, setArgs] = useState((server.args ?? []).join(" "));
  const [saving, setSaving] = useState(false);

  const [autoStart, setAutoStart] = useState(server.auto_start);
  const hasLocalProcess = !!server.command || server.auto_start;

  const handleSave = async () => {
    setSaving(true);
    try {
      const parsedPort = port.trim() ? parseInt(port.trim(), 10) : null;
      const cleanCommand = command.trim().replace(/^["']+|["']+$/g, "");
      await api.updateMCPServer(server.id, {
        name,
        transport,
        url: transport !== "stdio" ? url : null,
        port: (hasLocalProcess && parsedPort) ? parsedPort : null,
        command: (transport === "stdio" || hasLocalProcess) ? cleanCommand || null : null,
        args: (transport === "stdio" || hasLocalProcess) ? args.split(/\s+/).filter(Boolean) : [],
        auto_start: autoStart,
      });
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-lg p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">Edit MCP Server</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>
        <div className="space-y-4">
          <Field label="Name" value={name} onChange={setName} placeholder="My MCP Server" />
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Transport</label>
            <select className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 text-sm" value={transport} onChange={(e) => setTransport(e.target.value)}>
              <option value="streamable_http">Streamable HTTP</option>
              <option value="stdio">stdio (local command)</option>
              <option value="sse">SSE (legacy)</option>
            </select>
          </div>
          {transport !== "stdio" && !hasLocalProcess && <Field label="URL" value={url} onChange={setUrl} placeholder="http://localhost:3000/mcp" />}
          {hasLocalProcess && (
            <div>
              <label className="block text-sm font-medium text-th-text-tertiary mb-2">Port</label>
              <input
                type="number"
                className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-all text-sm"
                value={port}
                onChange={(e) => setPort(e.target.value)}
                placeholder="5000"
                min={1}
                max={65535}
              />
              <p className="text-[11px] text-th-text-muted mt-1">Used for both the executable --port arg and the MCP client connection</p>
            </div>
          )}
          {(transport === "stdio" || hasLocalProcess) && (<><Field label="Command (no quotes needed — spaces in paths are OK)" value={command} onChange={setCommand} placeholder="Auto-detected if left empty" /><Field label="Arguments (space-separated)" value={args} onChange={setArgs} placeholder="run testserver" /></>)}
          {hasLocalProcess && (
            <label className="flex items-center gap-3 cursor-pointer">
              <input type="checkbox" checked={autoStart} onChange={(e) => setAutoStart(e.target.checked)} className="w-4 h-4 rounded border-th-border bg-th-input-bg text-th-text-primary focus:ring-blue-400" />
              <span className="text-sm text-th-text-tertiary">Auto-start process on backend startup</span>
            </label>
          )}
        </div>
        <div className="flex justify-end gap-2 mt-6 pt-4 border-t border-th-border">
          <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
          <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40" onClick={handleSave} disabled={!name || saving}>{saving ? "Saving..." : "Save"}</button>
        </div>
      </div>
    </div>
  );
}

function ImportMCPDialog({ onClose, onImported }: { onClose: () => void; onImported: () => void }) {
  const [jsonText, setJsonText] = useState("");
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<{ added: string[]; skipped: string[] } | null>(null);
  const [error, setError] = useState("");

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setJsonText(ev.target?.result as string ?? "");
    reader.readAsText(file);
  };

  const handleImport = async () => {
    setError("");
    setResult(null);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e: unknown) {
      setError("Invalid JSON: " + (e instanceof SyntaxError ? e.message : String(e)));
      return;
    }
    if (!parsed.mcpServers && !parsed.mcp_servers) {
      setError('JSON must contain an "mcpServers" key');
      return;
    }
    setImporting(true);
    try {
      const res = await api.importMCPServers(parsed);
      setResult(res);
      if (res.added.length > 0) setTimeout(onImported, 1500);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setImporting(false);
    }
  };

  const placeholder = `{
  "mcpServers": {
    "My Server": {
      "command": "npx",
      "args": ["-y", "@example/mcp-server"]
    },
    "Remote Server": {
      "url": "http://localhost:3000/mcp"
    }
  }
}`;

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-xl p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">Import MCP Servers</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>

        <p className="text-xs text-th-text-tertiary mb-3">
          Paste JSON or upload a file. Uses the standard <code className="text-th-text-tertiary">mcpServers</code> format (same as Claude Desktop config).
        </p>

        <textarea
          className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all font-mono text-xs min-h-[220px]"
          value={jsonText}
          onChange={(e) => setJsonText(e.target.value)}
          placeholder={placeholder}
          spellCheck={false}
        />

        <div className="mt-3 flex items-center gap-3">
          <label className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5">
            <Upload size={12} /> Upload .json
            <input type="file" accept=".json,application/json" className="hidden" onChange={handleFileUpload} />
          </label>
        </div>

        {error && <p className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}

        {result && (
          <div className="mt-3 text-xs rounded-lg px-3 py-2 bg-th-inset-bg border border-th-border space-y-1">
            {result.added.length > 0 && <p className="text-emerald-400">Added: {result.added.join(", ")}</p>}
            {result.skipped.length > 0 && <p className="text-th-text-tertiary">Skipped (already exist): {result.skipped.join(", ")}</p>}
            {result.added.length === 0 && result.skipped.length === 0 && <p className="text-th-text-tertiary">No servers found in JSON</p>}
          </div>
        )}

        <div className="flex justify-end gap-2 mt-5 pt-4 border-t border-th-border">
          <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
          <button
            className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40"
            onClick={handleImport}
            disabled={!jsonText.trim() || importing}
          >
            {importing ? "Importing..." : "Import"}
          </button>
        </div>
      </div>
    </div>
  );
}

function EditJsonDialog({ servers, onClose, onSaved }: { servers: MCPServerStatus[]; onClose: () => void; onSaved: () => void }) {
  const buildJson = () => {
    const obj: Record<string, Record<string, unknown>> = {};
    for (const srv of servers) {
      if (srv.builtin) continue;
      const entry: Record<string, unknown> = {};
      if (srv.transport === "stdio") {
        entry.command = srv.command ?? "";
        if (srv.args?.length) entry.args = srv.args;
      } else {
        entry.url = srv.url ?? "";
        if (srv.transport !== "streamable_http") entry.transport = srv.transport;
      }
      obj[srv.name] = entry;
    }
    return JSON.stringify({ mcpServers: obj }, null, 2);
  };

  const [jsonText, setJsonText] = useState(buildJson);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [details, setDetails] = useState<string[]>([]);

  const handleSave = async () => {
    setError("");
    setDetails([]);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e: unknown) {
      setError("Invalid JSON: " + (e instanceof SyntaxError ? e.message : String(e)));
      return;
    }
    if (!parsed.mcpServers && !parsed.mcp_servers) {
      setError('JSON must contain an "mcpServers" key');
      return;
    }
    setSaving(true);
    try {
      const res = await api.saveMCPServersJson(parsed);
      if (res.error) {
        setError(res.error);
        setDetails(res.details ?? []);
      } else {
        onSaved();
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const nonBuiltinCount = servers.filter((s) => !s.builtin).length;

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-2xl p-6 shadow-2xl max-h-[85vh] flex flex-col">
        <div className="flex items-center justify-between mb-4 shrink-0">
          <div>
            <h2 className="text-lg font-bold text-th-text-primary">Edit MCP Servers (JSON)</h2>
            <p className="text-xs text-th-text-tertiary mt-1">{nonBuiltinCount} user-defined server{nonBuiltinCount !== 1 ? "s" : ""}. Managed servers are not included.</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>

        <textarea
          className="flex-1 w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all font-mono text-xs min-h-[300px] resize-y"
          value={jsonText}
          onChange={(e) => setJsonText(e.target.value)}
          spellCheck={false}
        />

        {error && (
          <div className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 shrink-0">
            <p className="font-semibold">{error}</p>
            {details.length > 0 && <ul className="mt-1 list-disc list-inside space-y-0.5 text-red-400/80">{details.map((d, i) => <li key={i}>{d}</li>)}</ul>}
          </div>
        )}

        <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-th-border shrink-0">
          <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
          <button
            className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40"
            onClick={handleSave}
            disabled={!jsonText.trim() || saving}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Dialog for managing the credentials a generated (or imported stdio)
// MCP server needs in its subprocess env.  All operations route
// through ``/api/vault/...`` which talks to the OS keychain — values
// never round-trip through the LLM context, never appear in the
// servers list, never get logged.
//
// We deliberately re-fetch the list of stored names on open instead of
// trusting ``srv.required_secrets`` alone: the user might have
// previously stored an EXTRA secret that isn't strictly required (e.g.
// optional STRIPE_WEBHOOK_SECRET on top of STRIPE_SECRET_KEY), and we
// want them to be able to clear it.
function CredentialsDialog({
  server,
  onClose,
  onChanged,
}: {
  server: MCPServerStatus;
  onClose: () => void;
  onChanged: () => Promise<void> | void;
}) {
  const [storedNames, setStoredNames] = useState<string[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [reveal, setReveal] = useState<Record<string, boolean>>({});
  const [savingName, setSavingName] = useState<string | null>(null);
  const [deletingName, setDeletingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshNames = async () => {
    try {
      const r = await api.vaultListNames(server.id);
      setStoredNames(r.names);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load credentials");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshNames();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [server.id]);

  // Build the union of declared-required + declared-optional +
  // already-stored names so the user sees every slot they could set.
  // ``server.required_secrets`` is authoritative for what's *needed*
  // to start; ``server.optional_secrets`` are personalisation slots
  // (the MCP has a default); ``storedNames`` is authoritative for
  // what's actually in the keychain.
  const optionalNames = server.optional_secrets ?? [];
  const allNames = Array.from(
    new Set([...server.required_secrets, ...optionalNames, ...storedNames])
  ).sort();

  const handleSave = async (name: string) => {
    const value = (drafts[name] ?? "").trim();
    if (!value) {
      setError(`Refusing to store empty value for ${name}.`);
      return;
    }
    setError(null);
    setSavingName(name);
    try {
      await api.vaultSetSecret(server.id, name, value);
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      setReveal((prev) => ({ ...prev, [name]: false }));
      await refreshNames();
      await onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to store credential");
    } finally {
      setSavingName(null);
    }
  };

  const handleDelete = async (name: string) => {
    setError(null);
    setDeletingName(name);
    try {
      await api.vaultDeleteSecret(server.id, name);
      await refreshNames();
      await onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete credential");
    } finally {
      setDeletingName(null);
    }
  };

  const interactiveAuth = server.auth && server.auth.kind !== "static";

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-xl p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-bold text-th-text-primary flex items-center gap-2">
              <KeyRound size={18} className="text-th-text-tertiary" />
              Credentials — {server.name}
            </h2>
            <p className="text-xs text-th-text-tertiary mt-1">
              Stored in your OS keychain. Values are injected into the MCP subprocess
              environment at start time and never returned through the API.
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>

        {error && (
          <div className="mb-4 flex items-center justify-between gap-3 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-xs text-red-400">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400/60 hover:text-red-400"><X size={14} /></button>
          </div>
        )}

        {interactiveAuth && server.auth && (
          <InteractiveAuthPanel
            serverId={server.id}
            initialStatus={server.auth}
            onChanged={onChanged}
          />
        )}

        {loading ? (
          <div className="flex items-center justify-center py-10 text-th-text-muted">
            <Loader2 size={20} className="animate-spin" />
          </div>
        ) : allNames.length === 0 ? (
          interactiveAuth ? null : (
            <div className="py-8 text-center text-sm text-th-text-tertiary">
              This server doesn't declare any required credentials.
            </div>
          )
        ) : (
          <div className="space-y-3 max-h-[55vh] overflow-y-auto pr-1">
            {allNames.map((name) => {
              const isStored = storedNames.includes(name);
              const isRequired = server.required_secrets.includes(name);
              const isOptional = optionalNames.includes(name);
              const draft = drafts[name] ?? "";
              const revealed = reveal[name] ?? false;
              return (
                <div key={name} className="bg-th-inset-bg border border-th-border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <code className="text-xs font-mono font-semibold text-th-text-primary">{name}</code>
                      {isRequired ? (
                        <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold">required</span>
                      ) : isOptional ? (
                        <span
                          className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold"
                          title="The MCP has a working default; setting a value here personalises it."
                        >
                          optional
                        </span>
                      ) : (
                        <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-th-card-bg text-th-text-muted border border-th-border font-semibold">extra</span>
                      )}
                      {isStored ? (
                        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 size={9} /> stored
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                          <XCircle size={9} /> missing
                        </span>
                      )}
                    </div>
                    {isStored && (
                      <button
                        type="button"
                        className="text-[11px] px-2 py-1 rounded bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-50 flex items-center gap-1"
                        onClick={() => handleDelete(name)}
                        disabled={deletingName === name}
                      >
                        {deletingName === name ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
                        Forget
                      </button>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type={revealed ? "text" : "password"}
                      autoComplete="off"
                      spellCheck={false}
                      placeholder={isStored ? "Enter new value to overwrite…" : "Paste value to store"}
                      className="flex-1 px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-xs font-mono"
                      value={draft}
                      onChange={(e) => setDrafts((prev) => ({ ...prev, [name]: e.target.value }))}
                    />
                    <button
                      type="button"
                      onClick={() => setReveal((prev) => ({ ...prev, [name]: !revealed }))}
                      className="p-2 rounded-lg bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary"
                      title={revealed ? "Hide" : "Show"}
                    >
                      {revealed ? <EyeOff size={12} /> : <Eye size={12} />}
                    </button>
                    <button
                      type="button"
                      className="px-3 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-xs font-semibold transition-opacity hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
                      onClick={() => handleSave(name)}
                      disabled={!draft.trim() || savingName === name}
                    >
                      {savingName === name ? <Loader2 size={12} className="animate-spin" /> : <KeyRound size={12} />}
                      Save
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="flex justify-end gap-2 mt-5 pt-4 border-t border-th-border">
          <button
            className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors"
            onClick={onClose}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}


// Renders the Login / Re-login / Logout affordance for any MCP whose
// auth.kind is NOT "static" (OAuth device, OAuth auth-code,
// browser-capture).  Lives inside the credentials dialog but is its own
// component so its login state is local — the dialog can still display
// the legacy paste-a-string row list below for any "extra" credentials
// the same MCP also wants (e.g. a tenant id env var alongside an OAuth
// bearer token).
function InteractiveAuthPanel({
  serverId,
  initialStatus,
  onChanged,
}: {
  serverId: string;
  initialStatus: MCPAuthStatus;
  onChanged: () => Promise<void> | void;
}) {
  const [status, setStatus] = useState<MCPAuthStatus>(initialStatus);
  const [busy, setBusy] = useState<"login" | "logout" | null>(null);
  const [panelError, setPanelError] = useState<string | null>(null);

  const refreshStatus = async () => {
    try {
      setStatus(await api.mcpAuthStatus(serverId));
    } catch {
      // Keep the last status — surfacing a polling error is noisier
      // than helpful when the user is mid-flow.
    }
  };

  const handleLogin = async () => {
    setPanelError(null);
    setBusy("login");
    try {
      const result = await api.mcpAuthLogin(serverId);
      if (result.status !== "ok") {
        setPanelError(result.message ?? `Login failed (${result.reason ?? result.auth_kind ?? "unknown"})`);
      } else if (result.auth) {
        setStatus(result.auth);
      } else {
        await refreshStatus();
      }
      await onChanged();
    } catch (e) {
      setPanelError(e instanceof Error ? e.message : "Login request failed");
    } finally {
      setBusy(null);
    }
  };

  const handleLogout = async () => {
    setPanelError(null);
    setBusy("logout");
    try {
      const result = await api.mcpAuthLogout(serverId);
      setStatus(result.auth);
      await onChanged();
    } catch (e) {
      setPanelError(e instanceof Error ? e.message : "Logout failed");
    } finally {
      setBusy(null);
    }
  };

  const kindLabel = AUTH_KIND_LABEL[status.kind] ?? status.kind;
  const expiry = status.expiry_iso ? new Date(status.expiry_iso) : null;
  const expiryLabel = expiry && !Number.isNaN(expiry.getTime())
    ? expiry.toLocaleString()
    : null;

  return (
    <div className="mb-4 p-4 bg-th-inset-bg border border-th-border rounded-lg">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <KeyRound size={14} className="text-th-text-tertiary" />
            <span className="text-xs font-semibold uppercase tracking-wider text-th-text-tertiary">
              {kindLabel}
            </span>
            {status.has_bundle ? (
              status.expired ? (
                <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  <XCircle size={9} /> expired
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <CheckCircle2 size={9} /> signed in
                </span>
              )
            ) : (
              <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <XCircle size={9} /> not signed in
              </span>
            )}
          </div>
          <p className="text-xs text-th-text-tertiary">
            {status.has_bundle && !status.expired
              ? expiryLabel
                ? `Token valid until ${expiryLabel}.`
                : "Token stored — expiry unknown."
              : status.needs_login
                ? "Click Login to open your browser and complete the sign-in flow."
                : "A login is required before this MCP can start."}
          </p>
        </div>
        <div className="flex flex-col gap-2 shrink-0">
          <button
            type="button"
            onClick={handleLogin}
            disabled={busy !== null}
            className="px-3 py-1.5 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-xs font-semibold transition-opacity hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
          >
            {busy === "login" ? <Loader2 size={12} className="animate-spin" /> : <LogIn size={12} />}
            {status.has_bundle ? "Re-login" : "Login"}
          </button>
          {status.has_bundle && (
            <button
              type="button"
              onClick={handleLogout}
              disabled={busy !== null}
              className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors disabled:opacity-40 flex items-center gap-1.5"
            >
              {busy === "logout" ? <Loader2 size={12} className="animate-spin" /> : <LogOut size={12} />}
              Logout
            </button>
          )}
        </div>
      </div>
      {panelError && (
        <div className="mt-3 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded text-xs text-red-400 flex items-center justify-between gap-3">
          <span className="break-words">{panelError}</span>
          <button onClick={() => setPanelError(null)} className="text-red-400/60 hover:text-red-400 shrink-0"><X size={12} /></button>
        </div>
      )}
    </div>
  );
}


function ToolChips({ srv, onToggle }: { srv: MCPServerStatus; onToggle: (serverId: string, tool: string, exclude: boolean) => void }) {
  if (!srv.connected || srv.tools.length === 0) return null;
  const excluded = new Set(srv.excluded_tools ?? []);
  const enabledCount = srv.tools.filter((t) => !excluded.has(t)).length;
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Tools</span>
        <span className="text-[10px] text-th-text-muted">{enabledCount}/{srv.tools.length} enabled</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {srv.tools.map((tool) => {
          const isExcluded = excluded.has(tool);
          return (
            <button
              key={tool}
              onClick={() => onToggle(srv.id, tool, !isExcluded)}
              className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md font-medium transition-all cursor-pointer ${
                isExcluded
                  ? "bg-th-code-bg border border-th-border/60 text-th-text-muted line-through opacity-50 hover:opacity-75"
                  : "bg-th-inset-bg border border-th-border text-th-text-tertiary hover:border-th-border-strong hover:text-th-text-secondary"
              }`}
              title={isExcluded ? `Enable ${tool}` : `Disable ${tool}`}
            >
              <Wrench size={10} className={isExcluded ? "text-th-text-secondary" : "text-th-text-muted"} />
              {tool}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (<div><label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label><input className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-all text-sm" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} /></div>);
}
