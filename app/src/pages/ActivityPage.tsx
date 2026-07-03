import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity as ActivityIcon,
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  ExternalLink,
  FileText,
  Loader2,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { api, type ActivityRow } from "../hooks/useApi";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

function todayIso(): string {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return localIso(d);
}

function localIso(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function shiftIso(iso: string, days: number): string {
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  dt.setDate(dt.getDate() + days);
  return localIso(dt);
}

function formatDateLabel(iso: string): string {
  const today = todayIso();
  if (iso === today) return "Today";
  if (iso === shiftIso(today, -1)) return "Yesterday";
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  const sameYear = dt.getFullYear() === new Date().getFullYear();
  return dt.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: sameYear ? undefined : "numeric",
  });
}

/**
 * Best-effort parser for human-typed dates. Accepts: ISO (`2026-05-05`),
 * slash dates (`5/5`, `5/5/26`, `05/05/2026`), "today" / "yesterday",
 * and natural-language fallbacks via `new Date(...)` (e.g. `May 5`).
 * Returns a local YYYY-MM-DD or `null` if it can't make sense of the input.
 */
function parseDateInput(raw: string): string | null {
  const s = raw.trim().toLowerCase();
  if (!s) return null;
  if (s === "today" || s === "now") return todayIso();
  if (s === "yesterday") return shiftIso(todayIso(), -1);
  if (s === "tomorrow") return shiftIso(todayIso(), 1);

  const buildIso = (y: number, m: number, d: number): string | null => {
    const dt = new Date(y, m - 1, d);
    if (dt.getFullYear() !== y || dt.getMonth() !== m - 1 || dt.getDate() !== d) return null;
    return localIso(dt);
  };

  // YYYY-MM-DD or YYYY/MM/DD
  const iso = s.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$/);
  if (iso) return buildIso(+iso[1], +iso[2], +iso[3]);

  // M/D or M/D/YYYY (locale-ambiguous, but consistent with the rest of the app's en-US bias)
  const slash = s.match(/^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$/);
  if (slash) {
    let y = slash[3] ? +slash[3] : new Date().getFullYear();
    if (y < 100) y += y >= 70 ? 1900 : 2000;
    return buildIso(y, +slash[1], +slash[2]);
  }

  // Natural language fallback ("May 5", "5 May 2026", "March 1st").
  // Append the current year if it looks like a bare month/day.
  const candidate = /\d{4}/.test(s) ? raw : `${raw} ${new Date().getFullYear()}`;
  const dt = new Date(candidate);
  if (!isNaN(dt.getTime())) return localIso(dt);

  return null;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

interface Status {
  enabled: boolean;
  interval_secs: number;
  retain_days: number;
  exclude_apps: string[];
  running: boolean;
  db_size_bytes: number;
  max_db_mb: number;
}

const PAGE_SIZE = 20;

// App color chip — deterministic hue from app name
function appHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
  return h % 360;
}

function AppBadge({ name }: { name: string }) {
  const hue = appHue(name);
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0"
      style={{
        background: `hsl(${hue},55%,18%)`,
        color: `hsl(${hue},80%,72%)`,
        border: `1px solid hsl(${hue},55%,28%)`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full shrink-0"
        style={{ background: `hsl(${hue},75%,55%)` }}
      />
      {name}
    </span>
  );
}

function ActivityTableRow({ r, expanded, onToggle }: { r: ActivityRow; expanded: boolean; onToggle: () => void }) {
  const hasDetail = !!(r.context || r.file_path || r.url);
  return (
    <>
      <tr
        className={`group transition-colors ${hasDetail ? "cursor-pointer" : ""} hover:bg-th-surface-hover/30 ${expanded ? "bg-th-surface-hover/20" : ""}`}
        onClick={hasDetail ? onToggle : undefined}
      >
        {/* Time */}
        <td className="pl-4 pr-4 py-2.5 text-xs font-mono text-th-text-muted whitespace-nowrap w-[80px]">
          {formatTime(r.ts)}
        </td>
        {/* App */}
        <td className="px-2 py-2.5 whitespace-nowrap w-[160px]">
          <AppBadge name={r.app} />
        </td>
        {/* Title */}
        <td className="px-2 py-2.5 min-w-0 max-w-0">
          <div className="flex flex-col min-w-0">
            {r.title && (
              <span className="text-xs text-th-text-primary truncate">{r.title}</span>
            )}
            {!r.title && !r.url && !r.file_path && (
              <span className="text-xs text-th-text-muted italic">—</span>
            )}
            {r.url && (
              <a
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-1 text-[11px] text-sky-400 hover:text-sky-300 truncate mt-0.5"
              >
                <ExternalLink size={9} className="shrink-0" />
                <span className="truncate">{r.url}</span>
              </a>
            )}
            {r.file_path && (
              <span className="inline-flex items-center gap-1 text-[11px] text-th-text-tertiary font-mono truncate mt-0.5">
                <FileText size={9} className="shrink-0" />
                <span className="truncate" title={r.file_path}>{r.file_path}</span>
              </span>
            )}
          </div>
        </td>
        {/* Duration */}
        <td className="px-2 py-2.5 text-right whitespace-nowrap w-[64px]">
          {r.duration_s > 0 ? (
            <span className="text-[11px] text-th-text-muted tabular-nums">{formatDuration(r.duration_s)}</span>
          ) : null}
        </td>
        {/* Expand toggle */}
        <td className="pl-2 pr-4 py-2.5 w-6">
          {hasDetail && (
            <ChevronDown
              size={13}
              className={`text-th-text-muted transition-transform ${expanded ? "rotate-180" : ""} group-hover:text-th-text-secondary`}
            />
          )}
        </td>
      </tr>
      {expanded && hasDetail && (
        <tr className="bg-th-surface-hover/10">
          <td colSpan={5} className="px-4 pb-3 pt-0">
            <div className="ml-[80px] pl-4 border-l-2 border-th-border/40 space-y-2">
              {r.context && (
                <p className="text-[11px] text-th-text-secondary whitespace-pre-line break-words leading-relaxed">
                  {r.context}
                </p>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function PaginationBar({
  page,
  totalPages,
  total,
  loading,
  onPage,
}: {
  page: number;
  totalPages: number;
  total: number;
  loading: boolean;
  onPage: (p: number) => void;
}) {
  if (total <= PAGE_SIZE) return null;
  return (
    <div className="flex items-center gap-0.5">
      <button
        type="button"
        onClick={() => onPage(0)}
        disabled={page === 0 || loading}
        className="px-1.5 py-0.5 rounded text-[12px] leading-none text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        title="First page"
      >«</button>
      <button
        type="button"
        onClick={() => onPage(page - 1)}
        disabled={page === 0 || loading}
        className="px-1.5 py-0.5 rounded text-[12px] leading-none text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        title="Previous page"
      >‹</button>
      <span className="px-2 text-[11px] text-th-text-secondary tabular-nums select-none">
        {page + 1} / {totalPages}
      </span>
      <button
        type="button"
        onClick={() => onPage(page + 1)}
        disabled={page >= totalPages - 1 || loading}
        className="px-1.5 py-0.5 rounded text-[12px] leading-none text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        title="Next page"
      >›</button>
      <button
        type="button"
        onClick={() => onPage(totalPages - 1)}
        disabled={page >= totalPages - 1 || loading}
        className="px-1.5 py-0.5 rounded text-[12px] leading-none text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        title="Last page"
      >»</button>
    </div>
  );
}

function ActivityTable({
  rows,
  expandedIds,
  onToggle,
}: {
  rows: ActivityRow[];
  expandedIds: Set<number>;
  onToggle: (id: number) => void;
}) {
  return (
    <table className="w-full border-collapse text-left table-fixed">
      <colgroup>
        <col style={{ width: "80px" }} />
        <col style={{ width: "160px" }} />
        <col style={{ minWidth: 0 }} />
        <col style={{ width: "64px" }} />
        <col style={{ width: "24px" }} />
      </colgroup>
      <thead>
        <tr className="border-b border-th-border/50">
          <th className="pl-4 pr-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">Time</th>
          <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">App</th>
          <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">Title / URL / File</th>
          <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted text-right">Duration</th>
          <th className="pl-2 pr-4 py-2" />
        </tr>
      </thead>
      <tbody className="divide-y divide-th-border/20">
        {rows.map((r) => (
          <ActivityTableRow
            key={r.id}
            r={r}
            expanded={expandedIds.has(r.id)}
            onToggle={() => onToggle(r.id)}
          />
        ))}
      </tbody>
    </table>
  );
}

export default function ActivityPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [date, setDate] = useState(todayIso());
  const [search, setSearch] = useState("");
  const dateInputRef = useRef<HTMLInputElement | null>(null);
  const [dateDraft, setDateDraft] = useState<string>(formatDateLabel(todayIso()));
  const [dateInvalid, setDateInvalid] = useState(false);

  useEffect(() => {
    setDateDraft(formatDateLabel(date));
    setDateInvalid(false);
    setPage(0);
  }, [date]);
  const [rows, setRows] = useState<ActivityRow[]>([]);
  const [appSummary, setAppSummary] = useState<{ app: string; seconds: number }[]>([]);
  const [totalSeconds, setTotalSeconds] = useState(0);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  // Zero-based page index shared by both Day and Search modes.
  const [page, setPage] = useState(0);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const toggleExpanded = useCallback((id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  // Whether the user has typed a search query.
  const searchMode = search.trim().length > 0;

  // Race-guard token.
  const fetchToken = useRef(0);

  // Reset to page 0 whenever the date or search query changes.
  useEffect(() => { setPage(0); }, [date, search]);

  // Debounced search query.
  const [debouncedSearch, setDebouncedSearch] = useState(search);
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search), 220);
    return () => clearTimeout(id);
  }, [search]);

  const refresh = useCallback(async () => {
    const token = ++fetchToken.current;
    setLoading(true);
    try {
      const st = await api.getActivityStatus();
      if (token !== fetchToken.current) return;
      setStatus(st);

      const q = debouncedSearch.trim();
      if (q) {
        // Search mode — server-side page (20 per request, replace not append).
        const res = await api.searchActivity({
          q,
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
          order_by: "rank",
        });
        if (token !== fetchToken.current) return;
        setRows(res.rows);
        setTotal(res.total);
        setAppSummary([]);
        setTotalSeconds(0);
      } else {
        // Day mode — fetch full day for summary, then slice client-side.
        const res = await api.getActivityTimeline(date);
        if (token !== fetchToken.current) return;
        setRows(res.rows);
        setTotal(res.rows.length);
        setAppSummary(res.summary.apps);
        setTotalSeconds(res.summary.total_seconds);
      }
    } catch (e) {
      console.warn("[Activity] refresh failed:", e);
    } finally {
      if (token === fetchToken.current) setLoading(false);
    }
  }, [date, debouncedSearch, page]);

  useEffect(() => { refresh(); }, [refresh]);

  // Live-poll every 15s in Day mode on today's date only.
  useEffect(() => {
    if (searchMode) return;
    if (date !== todayIso()) return;
    const id = setInterval(() => { refresh(); }, 15_000);
    return () => clearInterval(id);
  }, [date, refresh, searchMode]);

  const handleClear = async () => {
    try {
      await api.clearActivity();
      setConfirmClear(false);
      refresh();
    } catch (e) {
      console.warn("[Activity] clear failed:", e);
    }
  };

  // Day mode: slice the full day's rows to the current page window.
  // Search mode: server already returned PAGE_SIZE rows for this page.
  const filtered = useMemo(
    () => searchMode ? rows : rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE),
    [rows, page, searchMode],
  );
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Group rows by local date for the search-mode timeline render.
  // In Day mode every row is on the same date so we skip the grouping
  // and the divider chrome.
  const grouped = useMemo(() => {
    if (!searchMode) return null;
    const out: { date: string; rows: ActivityRow[] }[] = [];
    let current: { date: string; rows: ActivityRow[] } | null = null;
    for (const r of filtered) {
      const d = new Date(r.ts * 1000);
      const iso = localIso(d);
      if (!current || current.date !== iso) {
        current = { date: iso, rows: [] };
        out.push(current);
      }
      current.rows.push(r);
    }
    return out;
  }, [filtered, searchMode]);

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between gap-4 shrink-0 bg-th-bg-secondary">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-lg font-bold text-th-text-primary shrink-0">macOS Activity</h1>
          {status && (
            <span
              className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full font-medium ${
                status.enabled
                  ? status.running
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
                    : "bg-amber-500/15 text-amber-400 border border-amber-500/25"
                  : "bg-th-inset-bg text-th-text-muted border border-th-border"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  status.enabled
                    ? status.running
                      ? "bg-emerald-400 animate-pulse"
                      : "bg-amber-400"
                    : "bg-th-text-muted/50"
                }`}
              />
              {status.enabled ? (status.running ? "Tracking" : "Idle") : "Disabled"}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {(() => {
            const today = todayIso();
            const isToday = date === today;
            const goPrev = () => setDate((d) => shiftIso(d, -1));
            const goNext = () => {
              if (isToday) return;
              setDate((d) => {
                const next = shiftIso(d, 1);
                return next > today ? today : next;
              });
            };
            const openPicker = () => {
              const el = dateInputRef.current;
              if (!el) return;
              if (typeof el.showPicker === "function") {
                try { el.showPicker(); return; } catch { /* fallthrough */ }
              }
              el.focus();
              el.click();
            };
            const commitDraft = () => {
              const parsed = parseDateInput(dateDraft);
              if (parsed && parsed <= today) {
                setDate(parsed);
                setDateInvalid(false);
              } else if (dateDraft.trim() === "" || dateDraft === formatDateLabel(date)) {
                setDateInvalid(false);
              } else {
                setDateInvalid(true);
              }
            };
            return (
              <div className="inline-flex items-center gap-1.5">
              <div className={`inline-flex items-stretch rounded-lg border bg-th-input-bg overflow-hidden transition-colors ${dateInvalid ? "border-red-500/60" : "border-th-input-border focus-within:border-blue-400"}`}>
                <button
                  type="button"
                  onClick={goPrev}
                  className="px-2 flex items-center text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors border-r border-th-input-border"
                  title="Previous day"
                >
                  <ChevronLeft size={14} />
                </button>
                <div className="relative inline-flex items-center px-2.5 py-1.5 gap-2 min-w-[10rem]">
                  <button
                    type="button"
                    onClick={openPicker}
                    className="text-th-text-tertiary hover:text-th-text-primary transition-colors shrink-0"
                    title="Pick a date"
                    tabIndex={-1}
                  >
                    <Calendar size={13} />
                  </button>
                  <input
                    type="text"
                    value={dateDraft}
                    onChange={(e) => {
                      setDateDraft(e.target.value);
                      if (dateInvalid) setDateInvalid(false);
                    }}
                    onFocus={(e) => e.target.select()}
                    onBlur={commitDraft}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        commitDraft();
                        (e.target as HTMLInputElement).blur();
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        setDateDraft(formatDateLabel(date));
                        setDateInvalid(false);
                        (e.target as HTMLInputElement).blur();
                      }
                    }}
                    placeholder="YYYY-MM-DD"
                    aria-label="Date (type or pick)"
                    spellCheck={false}
                    className="bg-transparent outline-none text-sm font-medium text-th-text-primary w-full min-w-0 placeholder-th-text-muted"
                  />
                  <input
                    ref={dateInputRef}
                    type="date"
                    value={date}
                    onChange={(e) => e.target.value && setDate(e.target.value)}
                    max={today}
                    aria-hidden="true"
                    tabIndex={-1}
                    className="absolute left-0 bottom-0 w-px h-px opacity-0 pointer-events-none"
                  />
                </div>
                <button
                  type="button"
                  onClick={goNext}
                  disabled={isToday}
                  className="px-2 flex items-center text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors border-l border-th-input-border disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-th-text-tertiary disabled:cursor-not-allowed"
                  title={isToday ? "Already on today" : "Next day"}
                >
                  <ChevronRight size={14} />
                </button>
              </div>
              {!isToday && (
                <button
                  type="button"
                  onClick={() => setDate(today)}
                  className="px-2.5 py-1 text-[11px] font-medium text-sky-400 hover:bg-sky-500/10 rounded-lg border border-sky-500/25 transition-colors shrink-0"
                  title="Jump to today"
                >
                  Today
                </button>
              )}
              </div>
            );
          })()}
          <div className="relative max-w-xs w-full">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search apps, titles, URLs…"
              className="w-full pl-8 pr-8 py-1.5 rounded-lg bg-th-input-bg border border-th-input-border text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-colors"
            />
            {search && (
              <button onClick={() => setSearch("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-secondary transition-colors">
                <X size={14} />
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        {!status?.enabled && (
          <div className="m-6 p-4 rounded-xl border border-amber-500/20 bg-amber-500/5">
            <p className="text-sm text-amber-300 font-medium">Activity tracking is disabled.</p>
            <p className="text-xs text-th-text-tertiary mt-1">
              Enable it from <span className="font-mono text-th-text-secondary">Settings → macOS Activity</span> to start
              recording your local activity timeline. No screenshots are taken; only app names, window titles, and
              browser URLs are stored on your machine.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4 p-6">
          {/* Table */}
          <div className="min-w-0 bg-th-card-bg border border-th-card-border rounded-xl overflow-hidden">
            {loading && rows.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24">
                <Loader2 size={28} className="text-th-text-muted animate-spin mb-3" />
                <p className="text-sm text-th-text-tertiary">Loading activity…</p>
              </div>
            ) : filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24">
                <div className="w-16 h-16 rounded-2xl bg-th-inset-bg border border-th-border flex items-center justify-center mb-4">
                  <ActivityIcon size={28} className="text-th-text-muted" />
                </div>
                <p className="text-sm text-th-text-tertiary">
                  {searchMode ? "No matches for that search." : "No activity recorded for this day."}
                </p>
              </div>
            ) : searchMode && grouped ? (
              <>
                <div className="px-4 py-2.5 border-b border-th-border/50 flex items-center justify-between">
                  <span className="text-[11px] text-th-text-muted">
                    <span className="text-th-text-secondary font-medium">{total}</span>{" "}
                    {total === 1 ? "match" : "matches"} for{" "}
                    <span className="font-medium text-th-text-primary">"{search}"</span>
                  </span>
                  <PaginationBar page={page} totalPages={totalPages} total={total} loading={loading} onPage={setPage} />
                </div>
                {grouped.map((g) => (
                  <div key={g.date}>
                    <div className="px-4 py-2 bg-th-inset-bg/50 border-b border-th-border/30 flex items-center gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wide text-th-text-tertiary">
                        {formatDateLabel(g.date)}
                      </span>
                      <span className="text-[10px] text-th-text-muted tabular-nums">{g.rows.length} rows</span>
                    </div>
                    <ActivityTable rows={g.rows} expandedIds={expandedIds} onToggle={toggleExpanded} />
                  </div>
                ))}
              </>
            ) : (
              <>
                <div className="px-4 py-2.5 border-b border-th-border/30 flex items-center justify-between">
                  <span className="text-[11px] text-th-text-muted tabular-nums">
                    <span className="text-th-text-secondary font-medium">{total}</span>{" "}
                    {total === 1 ? "record" : "records"}
                  </span>
                  <PaginationBar page={page} totalPages={totalPages} total={total} loading={loading} onPage={setPage} />
                </div>
                <ActivityTable rows={filtered} expandedIds={expandedIds} onToggle={toggleExpanded} />
              </>
            )}
          </div>

          {/* Sidebar */}
          <aside className="space-y-4 lg:sticky lg:top-0 self-start">
            {searchMode ? (
              <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-th-text-primary">Search</h3>
                  <Search size={14} className="text-th-text-muted" />
                </div>
                <p className="text-2xl font-bold text-th-text-primary">{total}</p>
                <p className="text-[11px] text-th-text-muted mt-0.5">
                  {total === 1 ? "match" : "matches"} across all dates
                </p>
                <div className="mt-3 pt-3 border-t border-th-border/30 text-[11px] text-th-text-tertiary">
                  Ranked by relevance. Clear the search box to return to the daily timeline.
                </div>
              </div>
            ) : (
              <>
                <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-th-text-primary">Summary</h3>
                    <Clock size={14} className="text-th-text-muted" />
                  </div>
                  <p className="text-2xl font-bold text-th-text-primary">{formatDuration(totalSeconds)}</p>
                  <p className="text-[11px] text-th-text-muted mt-0.5">tracked on {formatDateLabel(date)}</p>
                </div>

                <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-th-text-primary mb-3">Top apps</h3>
                  {appSummary.length === 0 ? (
                    <p className="text-xs text-th-text-muted">No data yet.</p>
                  ) : (
                    <div className="space-y-2">
                      {appSummary.slice(0, 8).map((a) => {
                        const pct = totalSeconds > 0 ? (a.seconds / totalSeconds) * 100 : 0;
                        return (
                          <div key={a.app}>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-th-text-primary truncate">{a.app}</span>
                              <span className="text-th-text-muted shrink-0 ml-2">{formatDuration(a.seconds)}</span>
                            </div>
                            <div className="h-1 mt-1 rounded-full bg-th-inset-bg overflow-hidden">
                              <div className="h-full bg-blue-500/60" style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </>
            )}

            {status && (
              <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4 space-y-2 text-xs">
                <div className="flex justify-between text-th-text-tertiary">
                  <span>Polling</span>
                  <span className="text-th-text-secondary">every {status.interval_secs}s</span>
                </div>
                <div className="flex justify-between text-th-text-tertiary">
                  <span>Retention</span>
                  <span className="text-th-text-secondary">
                    {status.retain_days === 0 ? "forever" : `${status.retain_days} days`}
                  </span>
                </div>
                {(() => {
                  const hasCap = status.max_db_mb > 0;
                  const pct = hasCap
                    ? Math.min(100, (status.db_size_bytes / (status.max_db_mb * 1024 * 1024)) * 100)
                    : 0;
                  const barColor = pct > 85 ? "bg-red-500" : pct > 60 ? "bg-amber-500" : "bg-blue-500/70";
                  const capLabel = hasCap
                    ? (status.max_db_mb >= 1024
                      ? `${(status.max_db_mb / 1024).toFixed(1)} GB`
                      : `${status.max_db_mb} MB`)
                    : null;
                  return (
                    <div className="pt-1 pb-0.5 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-th-text-tertiary text-xs">Storage</span>
                        <span className="text-th-text-secondary text-xs tabular-nums font-medium">
                          {formatBytes(status.db_size_bytes)}
                          {capLabel && (
                            <span className="text-th-text-muted font-normal"> / {capLabel}</span>
                          )}
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-th-inset-bg border border-th-border/30 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${hasCap ? barColor : "bg-blue-500/40"}`}
                          style={{ width: hasCap ? `max(4px, ${pct}%)` : "100%" }}
                        />
                      </div>
                      {hasCap && (
                        <div className="flex justify-between text-[10px] text-th-text-muted tabular-nums">
                          <span>{pct.toFixed(1)}% used</span>
                          {pct > 85 && (
                            <span className="text-red-400 font-medium">Near limit</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })()}
                {status.exclude_apps.length > 0 && (
                  <div className="flex flex-col gap-1 pt-2 border-t border-th-border/40">
                    <span className="text-th-text-muted">Excluded apps</span>
                    <div className="flex flex-wrap gap-1">
                      {status.exclude_apps.map((a) => (
                        <span key={a} className="px-1.5 py-0.5 rounded text-[10px] bg-th-inset-bg text-th-text-secondary">
                          {a}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t border-th-border/40">
                  {confirmClear ? (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleClear}
                        className="px-2 py-1 rounded text-[11px] font-medium bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-all"
                      >
                        Confirm wipe
                      </button>
                      <button
                        onClick={() => setConfirmClear(false)}
                        className="px-2 py-1 rounded text-[11px] font-medium text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover transition-all"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmClear(true)}
                      className="w-full inline-flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-[11px] font-medium text-th-text-tertiary hover:text-red-400 hover:bg-red-500/10 transition-all"
                    >
                      <Trash2 size={11} />
                      Clear all activity data
                    </button>
                  )}
                </div>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
