import { useEffect, useRef, useState } from "react";

const THINKING_PHRASES = [
  "Thinking…",
  "On it…",
  "Let me think…",
  "Working through this…",
  "Connecting the dots…",
  "Reasoning step by step…",
  "Considering your request…",
  "Exploring possibilities…",
  "Putting the pieces together…",
  "Consulting the oracle…",
  "Deep in thought…",
  "Crunching…",
  "Untangling this…",
  "Spinning up the neurons…",
  "Weighing the options…",
  "Reading the context…",
  "Mapping it out…",
  "Chasing the answer…",
];

type StreamPhase = "thinking" | "memory_search";

interface ThinkingIndicatorProps {
  phase?: StreamPhase;
  pendingContext?: string[];
  /** When true, wraps with the bot avatar row used in ChatPage */
  withAvatar?: boolean;
}

const SNAKE_CELLS = 5;
const SNAKE_DURATION = 1800; // ms for one full forward pass
const SNAKE_DELAY_STEP = SNAKE_DURATION / SNAKE_CELLS;

const snakeKeyframes = `
@keyframes snake-pulse {
  0%, 15%    { opacity: 0.12; transform: scale(0.78); }
  35%, 65%   { opacity: 1;    transform: scale(1);    }
  85%, 100%  { opacity: 0.12; transform: scale(0.78); }
}
`;

function SnakeAnimation({ color }: { color: string }) {
  return (
    <>
      <style>{snakeKeyframes}</style>
      <div className="flex gap-[3px] items-center">
        {Array.from({ length: SNAKE_CELLS }).map((_, i) => (
          <span
            key={i}
            className={`inline-block w-[6px] h-[6px] rounded-[2px] ${color}`}
            style={{
              animation: `snake-pulse ${SNAKE_DURATION}ms ease-in-out infinite`,
              animationDelay: `${i * SNAKE_DELAY_STEP}ms`,
            }}
          />
        ))}
      </div>
    </>
  );
}

export function ThinkingIndicator({
  phase = "thinking",
  pendingContext = [],
  withAvatar = false,
}: ThinkingIndicatorProps) {
  const [phraseIdx, setPhraseIdx] = useState(() =>
    Math.floor(Math.random() * THINKING_PHRASES.length)
  );
  const phraseIdxRef = useRef(phraseIdx);
  phraseIdxRef.current = phraseIdx;

  useEffect(() => {
    if (pendingContext.length > 0 || phase !== "thinking") return;
    const id = setInterval(() => {
      let next: number;
      do {
        next = Math.floor(Math.random() * THINKING_PHRASES.length);
      } while (next === phraseIdxRef.current);
      setPhraseIdx(next);
    }, 2000);
    return () => clearInterval(id);
  }, [phase, pendingContext.length]);

  const isContext = pendingContext.length > 0;
  const isMemory = phase === "memory_search";

  const snakeColor = isContext
    ? "bg-violet-400"
    : isMemory
    ? "bg-blue-400"
    : "bg-th-text-muted/60";

  const textColor = isContext
    ? "text-violet-400/70"
    : isMemory
    ? "text-blue-400/70"
    : "text-th-text-muted";

  const label = isContext
    ? "Applying your context…"
    : isMemory
    ? "Searching memory…"
    : THINKING_PHRASES[phraseIdx];

  const inner = (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 h-7">
        <SnakeAnimation color={snakeColor} />
        <span className={`text-sm transition-opacity duration-300 ${textColor}`}>
          {label}
        </span>
      </div>
      {pendingContext.map((ctx, i) => (
        <div key={i} className="flex items-center gap-2 pl-0.5">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse shrink-0" />
          <span className="text-[11px] text-th-text-muted italic truncate max-w-xs">
            "{ctx.length > 60 ? ctx.slice(0, 60) + "…" : ctx}"
          </span>
        </div>
      ))}
    </div>
  );

  if (!withAvatar) return inner;

  return (
    <div className="flex gap-3 items-start">
      <div className="w-7 h-7 rounded-full border border-th-border/70 bg-th-inset-bg flex items-center justify-center shrink-0 mt-0.5">
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-th-text-muted"
        >
          <path d="M12 8V4H8" />
          <rect width="16" height="12" x="4" y="8" rx="2" />
          <path d="M2 14h2" />
          <path d="M20 14h2" />
          <path d="M15 13v2" />
          <path d="M9 13v2" />
        </svg>
      </div>
      {inner}
    </div>
  );
}
