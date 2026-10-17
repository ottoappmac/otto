import { memo } from "react";
import { MessageBubble, ThoughtBlock } from "./MessageBubble";
import { SubagentGroup } from "./SubagentGroup";
import type { ArtifactType } from "./ArtifactPanel";
import type { RenderItem } from "../../utils/renderItems";

// ---------------------------------------------------------------------------
// MessageList — memoized so that typing in the input does NOT trigger
// re-renders of the (potentially very long) message history. All callback
// props must be stable (useCallback) in the page that renders this.
// ---------------------------------------------------------------------------

interface MessageListProps {
  items: RenderItem[];
  thoughtFlags: boolean[];
  latestTodoFlags: boolean[];
  isStreaming: boolean;
  currentSessionId: string | null;
  resolveSubagentModel: (name: string) => { label: string; family: string };
  /** Omit to render user messages without the edit control. */
  onEditMessage?: (idx: number, content: string) => void;
  onHitlDecision: (id: string, decisions: Array<Record<string, unknown>>, requestId?: string) => void;
  onApproveAllSession?: (id: string, decisions: Array<Record<string, unknown>>) => void;
  onOpenArtifact: (path: string, fileUrl: string, type: ArtifactType) => void;
}

export const MessageList = memo(function MessageList({
  items,
  thoughtFlags,
  latestTodoFlags,
  isStreaming,
  currentSessionId,
  resolveSubagentModel,
  onEditMessage,
  onHitlDecision,
  onApproveAllSession,
  onOpenArtifact,
}: MessageListProps) {
  return (
    <>
      {items.map((item) => {
        if (item.kind === "subagent-group") {
          const model = resolveSubagentModel(item.name);
          return (
            <SubagentGroup
              key={`group-${item.name}`}
              name={item.name}
              messages={item.messages}
              modelLabel={model.label}
              modelFamily={model.family}
              sessionId={currentSessionId ?? undefined}
              onOpenArtifact={onOpenArtifact}
            />
          );
        }
        if (item.kind === "thought-group") {
          const streaming = item.messages.some((m) => m.metadata?.streaming);
          return (
            <ThoughtBlock
              key={`thought-${item.messages.map((m) => m.id).join("-")}-${streaming ? "live" : "done"}`}
              messages={item.messages}
            />
          );
        }
        const { message: msg, index: idx } = item;
        return (
          <MessageBubble
            key={msg.id}
            message={msg}
            isThought={thoughtFlags[idx]}
            isLatestTodo={latestTodoFlags[idx]}
            canEdit={Boolean(onEditMessage) && msg.type === "user" && !isStreaming}
            sessionId={currentSessionId ?? undefined}
            onEdit={onEditMessage ? (newContent) => onEditMessage(idx, newContent) : undefined}
            onHitlDecision={(decisions) => onHitlDecision(
              msg.id,
              decisions,
              typeof msg.metadata?.request_id === "string" ? msg.metadata.request_id : undefined,
            )}
            onApproveAllSession={onApproveAllSession ? (decisions) => onApproveAllSession(msg.id, decisions) : undefined}
            onOpenArtifact={onOpenArtifact}
          />
        );
      })}
    </>
  );
});
