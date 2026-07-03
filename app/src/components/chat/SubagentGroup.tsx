import { memo, useMemo, useState } from "react";
import { ChevronUp, ChevronRight, Cpu } from "lucide-react";
import { MessageBubble } from "./MessageBubble";
import { getToolLabel } from "../../utils/toolLabels";
import type { ChatMessage } from "../../types";
import { familyChipClasses } from "../../utils/subagentModelChip";
import { getAgentIcon } from "../../utils/entityIcons";
import type { ArtifactType } from "./ArtifactPanel";

interface SubagentGroupProps {
  name: string;
  messages: ChatMessage[];
  modelLabel?: string;
  modelFamily?: string;
  sessionId?: string;
  onOpenArtifact?: (path: string, fileUrl: string, type: ArtifactType) => void;
}

export const SubagentGroup = memo(function SubagentGroup({ name, messages, modelLabel, modelFamily, sessionId, onOpenArtifact }: SubagentGroupProps) {
  const [expanded, setExpanded] = useState(false);

  const { stepCount, hasRunning, latestLabel, latestTodoIndex } = useMemo(() => {
    let steps = 0;
    let running = false;
    let label = "";
    let todoIdx = -1;
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (m.type === "tool_call" || m.type === "tool_result") {
        steps++;
        if (m.type === "tool_call") running = true;
        label = getToolLabel(m.content, m.metadata?.args as Record<string, unknown> | undefined);
        if (m.content === "write_todos") todoIdx = i;
      }
    }
    return { stepCount: steps, hasRunning: running, latestLabel: label, latestTodoIndex: todoIdx };
  }, [messages]);

  return (
    <div className="ml-10">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs text-th-text-tertiary hover:text-th-text-secondary transition-colors py-1.5 group/tool"
      >
        {expanded ? <ChevronUp size={12} /> : <ChevronRight size={12} />}
        {(() => { const { Icon, className } = getAgentIcon("subagent"); return <Icon size={12} className={className} />; })()}
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium shrink-0">
          {name}
        </span>
        {modelLabel && (
          <span
            className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border font-medium shrink-0 ${familyChipClasses(modelFamily ?? "")}`}
            title={`Model: ${modelLabel}`}
            onClick={(e) => e.stopPropagation()}
          >
            <Cpu size={10} />
            <span className="truncate max-w-[180px]">{modelLabel}</span>
          </span>
        )}
        {latestLabel && (
          <span className="text-th-text-tertiary group-hover/tool:text-th-text-secondary truncate max-w-[300px]">
            {latestLabel}
          </span>
        )}
        <span className="text-th-text-muted shrink-0">
          {stepCount} {stepCount === 1 ? "step" : "steps"}
        </span>
        {hasRunning ? (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium animate-pulse shrink-0">
            running
          </span>
        ) : (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium shrink-0">
            done
          </span>
        )}
      </button>
      {expanded && (
        <div className="ml-4 border-l border-th-border pl-2">
          {messages.map((msg, i) => (
            <MessageBubble key={msg.id} message={msg} inGroup isLatestTodo={i === latestTodoIndex} sessionId={sessionId} onOpenArtifact={onOpenArtifact} />
          ))}
        </div>
      )}
    </div>
  );
});
