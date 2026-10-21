import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Tooltip } from "./Tooltip";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-th-text-primary text-th-bg hover:opacity-90",
  secondary: "border border-th-border bg-th-surface text-th-text-primary hover:bg-th-surface-hover",
  ghost: "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary",
  danger: "text-th-danger hover:bg-th-danger/10",
};
const SIZES: Record<Size, string> = {
  sm: "h-7 gap-1 px-2 text-xs",
  md: "h-8 gap-1.5 px-3 text-sm",
};
const BASE =
  "inline-flex shrink-0 items-center justify-center rounded-lg font-medium transition-colors duration-fast disabled:pointer-events-none disabled:opacity-40";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "sm", className = "", type = "button", ...rest },
  ref,
) {
  return <button ref={ref} type={type} className={`${BASE} ${VARIANTS[variant]} ${SIZES[size]} ${className}`} {...rest} />;
});

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  label: string;
  icon: ReactNode;
  keys?: string[];
  variant?: Variant;
  size?: Size;
}

/** Icon-only control: always has an accessible name and a fast tooltip. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, keys, variant = "ghost", size = "sm", className = "", type = "button", ...rest },
  ref,
) {
  const box = size === "sm" ? "h-7 w-7" : "h-8 w-8";
  return (
    <Tooltip label={label} keys={keys}>
      <button ref={ref} type={type} aria-label={label} className={`${BASE} ${VARIANTS[variant]} ${box} ${className}`} {...rest}>
        {icon}
      </button>
    </Tooltip>
  );
});
