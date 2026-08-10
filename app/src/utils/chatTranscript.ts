import type { ChatMessage } from "../types";
import { getToolLabel } from "./toolLabels";

/**
 * Render a chat session as markdown for the clipboard.
 *
 * Keeps the conversation itself verbatim (user turns, agent replies, errors)
 * and reduces each tool run to a one-line trace so the transcript stays
 * readable when pasted elsewhere.
 */
export function formatChatTranscript(messages: ChatMessage[]): string {
  const blocks: string[] = [];

  for (const msg of messages) {
    const subagent = msg.metadata?.subagent as string | undefined;
    const content = msg.content?.trim();

    switch (msg.type) {
      case "user": {
        if (!content) break;
        const heading = msg.metadata?.isContext ? "## You (added context)" : "## You";
        blocks.push(`${heading}\n\n${content}`);
        break;
      }
      case "agent": {
        if (!content) break;
        blocks.push(`## ${subagent ? `Otto — ${subagent}` : "Otto"}\n\n${content}`);
        break;
      }
      // A finished tool run is merged into a single "tool_result" entry whose
      // content is still the tool name (see mergeToolMessages).
      case "tool_call":
      case "tool_result": {
        const args = msg.metadata?.args as Record<string, unknown> | undefined;
        blocks.push(`_Tool: ${getToolLabel(msg.content, args)}_`);
        break;
      }
      case "ask_user": {
        if (content) blocks.push(`## Otto asked\n\n${content}`);
        break;
      }
      case "error": {
        if (content) blocks.push(`> **Error:** ${content}`);
        break;
      }
      default:
        break;
    }
  }

  return blocks.join("\n\n");
}
