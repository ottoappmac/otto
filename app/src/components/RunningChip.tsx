/**
 * RunningChip — a small pill indicator for a background process.
 *
 * Slides in on mount and out when dismissed. Visibility is controlled
 * by the parent; the chip never auto-dismisses.
 *
 * Pass `href` to make the label a navigation link.
 */
import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { X } from "lucide-react";

interface Props {
  icon: ReactNode;
  label: string;
  href?: string;
  onDismiss?: () => void;
}

export default function RunningChip({ icon, label, href, onDismiss }: Props) {
  const [visible, setVisible] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const tid = setTimeout(() => setVisible(true), 10);
    return () => clearTimeout(tid);
  }, []);

  const handleDismiss = () => {
    setVisible(false);
    setTimeout(() => onDismiss?.(), 300);
  };

  const handleNavigate = () => {
    if (href) navigate(href);
    handleDismiss();
  };

  return (
    <div
      className={`
        flex items-center gap-2 pl-3 pr-2 py-2
        bg-th-card-bg/90 border border-th-border rounded-full shadow-lg backdrop-blur
        transition-all duration-300 ease-out
        ${visible ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"}
      `}
    >
      {icon}
      {href ? (
        <button
          onClick={handleNavigate}
          className="text-xs text-th-text-secondary hover:text-th-text-primary whitespace-nowrap transition-colors"
        >
          {label}
        </button>
      ) : (
        <span className="text-xs text-th-text-secondary whitespace-nowrap">{label}</span>
      )}
      <button
        onClick={handleDismiss}
        className="p-0.5 rounded-full text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all shrink-0"
        title="Dismiss"
      >
        <X size={12} />
      </button>
    </div>
  );
}
