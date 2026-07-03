import { useNavigate, useLocation } from "react-router-dom";
import { WifiOff, ServerCrash, AlertTriangle, X } from "lucide-react";
import { useConnection } from "../context/ConnectionContext";

export function ConnectionBanner() {
  const { networkOnline, backendReachable, lastError, activeSessionId, clearError } = useConnection();
  const navigate = useNavigate();
  const location = useLocation();

  const isOnChatSession = activeSessionId && location.pathname === `/chat/${activeSessionId}`;

  if (networkOnline && backendReachable && !lastError) return null;

  let Icon = AlertTriangle;
  let message = "";
  let actionLabel = "";
  let onAction: (() => void) | null = null;
  let color = "amber";

  if (!networkOnline) {
    Icon = WifiOff;
    message = "You're offline. Check your internet connection.";
    color = "red";
  } else if (!backendReachable) {
    Icon = ServerCrash;
    message = "Lost connection to the backend server.";
    color = "red";
    actionLabel = "Retry";
    onAction = () => window.location.reload();
  } else if (lastError) {
    message = lastError.message;
    if (activeSessionId && !isOnChatSession) {
      actionLabel = "Resume Chat";
      onAction = () => {
        clearError();
        navigate(`/chat/${activeSessionId}`);
      };
    }
  }

  const palette: Record<string, { bg: string; border: string; text: string; icon: string; btnBg: string; btnHover: string; btnText: string }> = {
    amber: {
      bg: "bg-amber-50",
      border: "border-amber-200",
      text: "text-amber-800",
      icon: "text-amber-500",
      btnBg: "bg-amber-100",
      btnHover: "hover:bg-amber-200",
      btnText: "text-amber-700",
    },
    red: {
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-800",
      icon: "text-red-500",
      btnBg: "bg-red-100",
      btnHover: "hover:bg-red-200",
      btnText: "text-red-700",
    },
  };
  const c = palette[color];

  return (
    <div className={`flex items-center gap-3 px-4 py-2.5 ${c.bg} border-b ${c.border} ${c.text} text-sm shrink-0`}>
      <Icon size={16} className={c.icon} />
      <span className="flex-1">{message}</span>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className={`px-3 py-1 rounded-md ${c.btnBg} ${c.btnHover} ${c.btnText} text-xs font-medium transition-colors`}
        >
          {actionLabel}
        </button>
      )}
      {lastError && (
        <button
          onClick={clearError}
          className="p-0.5 rounded hover:bg-white/10 transition-colors"
          title="Dismiss"
        >
          <X size={14} />
        </button>
      )}
    </div>
  );
}
