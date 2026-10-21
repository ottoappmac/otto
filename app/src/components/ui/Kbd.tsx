import type { ReactNode } from "react";

/** A keycap, e.g. <Kbd>⌘</Kbd><Kbd>S</Kbd>. */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-[1.25rem] items-center justify-center rounded border border-th-border bg-th-inset-bg px-1 font-sans text-2xs leading-4 text-th-text-muted">
      {children}
    </kbd>
  );
}
