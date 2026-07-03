/**
 * NotificationCenter — bell button + dropdown inbox.
 *
 * Surfaces the persistent notification log from NotificationContext so the user
 * can always answer "what was that badge for". Each row deep-links to the
 * relevant event and marks itself read on click; the header offers
 * "Mark all read" and "Clear". Visual language mirrors AmbientToast.
 */
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bell,
  Calendar,
  Zap,
  MessageSquare,
  Sparkles,
  BrainCircuit,
  CheckCheck,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { useNotification, type NotificationItem } from "../context/NotificationContext";
import { formatRelativeTime } from "../utils/formatRelativeTime";

const CATEGORY_ICON: Record<NotificationItem["category"], LucideIcon> = {
  schedule: Calendar,
  trigger: Zap,
  session: MessageSquare,
  ambient: Sparkles,
  memory: BrainCircuit,
};

function kindColor(kind: NotificationItem["kind"]): string {
  if (kind === "error") return "text-red-400";
  if (kind === "hitl") return "text-amber-400";
  return "text-emerald-400";
}

function dotColor(kind: NotificationItem["kind"]): string {
  if (kind === "error") return "bg-red-500";
  if (kind === "hitl") return "bg-amber-400";
  return "bg-emerald-500";
}

interface Props {
  collapsed: boolean;
}

export default function NotificationCenter({ collapsed }: Props) {
  const navigate = useNavigate();
  const { items, unreadCount, markRead, markAllRead, clear, removeItem } = useNotification();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Close on outside click or Escape.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const handleRowClick = (item: NotificationItem) => {
    markRead(item.id);
    setOpen(false);
    navigate(item.deepLink);
  };

  return (
    <div ref={wrapperRef} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`relative flex items-center ${collapsed ? "justify-center" : "justify-center"} p-1 rounded text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all`}
        title={unreadCount > 0 ? `${unreadCount} unread notification${unreadCount === 1 ? "" : "s"}` : "Notifications"}
        aria-label="Notifications"
      >
        <Bell size={14} />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1.5 min-w-[16px] h-[16px] flex items-center justify-center rounded-full border-2 border-th-sidebar-bg bg-red-500 text-[9px] font-bold text-white px-0.5">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          className="absolute top-full right-0 mt-2 w-80 max-h-[60vh] flex flex-col bg-th-card-bg border border-th-border-strong rounded-xl shadow-2xl z-50 overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2.5 border-b border-th-sidebar-border/60 shrink-0">
            <p className="text-[13px] font-semibold text-th-text-primary">Notifications</p>
            <div className="flex items-center gap-1">
              <button
                onClick={markAllRead}
                disabled={unreadCount === 0}
                className="flex items-center gap-1 px-1.5 py-1 rounded text-[11px] text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-th-text-muted"
                title="Mark all read"
              >
                <CheckCheck size={13} />
              </button>
              <button
                onClick={clear}
                disabled={items.length === 0}
                className="flex items-center gap-1 px-1.5 py-1 rounded text-[11px] text-th-text-muted hover:text-red-400 hover:bg-red-500/10 transition-all disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-th-text-muted"
                title="Clear all"
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>

          {/* List */}
          <div className="overflow-y-auto flex-1 min-h-0">
            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 py-10 text-th-text-muted">
                <Bell size={20} className="opacity-50" />
                <p className="text-[12px]">No notifications</p>
              </div>
            ) : (
              <ul className="flex flex-col">
                {items.map((item) => {
                  const Icon = CATEGORY_ICON[item.category] ?? Bell;
                  return (
                    <li key={item.id} className="group relative">
                      <button
                        onClick={() => handleRowClick(item)}
                        className={`w-full text-left flex items-start gap-2.5 px-3 py-2.5 transition-all hover:bg-th-surface-hover ${
                          item.read ? "" : "bg-th-surface-hover/40"
                        }`}
                      >
                        <Icon size={15} className={`shrink-0 mt-0.5 ${kindColor(item.kind)}`} />
                        <div className="flex-1 min-w-0 pr-4">
                          <p className={`text-[12px] leading-snug ${item.read ? "text-th-text-secondary" : "text-th-text-primary font-medium"}`}>
                            {item.title}
                          </p>
                          {item.body && (
                            <p className="text-[11px] text-th-text-muted mt-0.5 line-clamp-2">{item.body}</p>
                          )}
                          <p className="text-[10px] text-th-text-muted mt-1">
                            {formatRelativeTime(new Date(item.createdAt).toISOString())}
                          </p>
                        </div>
                        {!item.read && (
                          <span className={`shrink-0 mt-1 w-2 h-2 rounded-full ${dotColor(item.kind)}`} />
                        )}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeItem(item.id);
                        }}
                        className="absolute top-2 right-2 p-0.5 rounded opacity-0 group-hover:opacity-100 text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all"
                        title="Remove"
                      >
                        <X size={12} />
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
