import { Loader2 } from "lucide-react";
import RunningChip from "../RunningChip";

interface Props {
  onDismiss?: () => void;
}

export default function AmbientSweepToast({ onDismiss }: Props) {
  return (
    <RunningChip
      icon={<Loader2 size={13} className="text-blue-400 animate-spin shrink-0" />}
      label="Generating suggestions…"
      href="/ambient"
      onDismiss={onDismiss}
    />
  );
}
