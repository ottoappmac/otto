/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        th: {
          bg: "rgb(var(--color-bg) / <alpha-value>)",
          "bg-secondary": "rgb(var(--color-bg-secondary) / <alpha-value>)",
          "bg-tertiary": "rgb(var(--color-bg-tertiary) / <alpha-value>)",
          surface: "rgb(var(--color-surface) / <alpha-value>)",
          "surface-hover": "rgb(var(--color-surface-hover) / <alpha-value>)",
          border: "rgb(var(--color-border) / <alpha-value>)",
          "border-strong": "rgb(var(--color-border-strong) / <alpha-value>)",
          "text-primary": "rgb(var(--color-text-primary) / <alpha-value>)",
          "text-secondary": "rgb(var(--color-text-secondary) / <alpha-value>)",
          "text-tertiary": "rgb(var(--color-text-tertiary) / <alpha-value>)",
          "text-muted": "rgb(var(--color-text-muted) / <alpha-value>)",
          "text-faint": "rgb(var(--color-text-faint) / <alpha-value>)",
          "sidebar-bg": "rgb(var(--color-sidebar-bg) / <alpha-value>)",
          "sidebar-border": "rgb(var(--color-sidebar-border) / <alpha-value>)",
          "input-bg": "rgb(var(--color-input-bg) / <alpha-value>)",
          "input-border": "rgb(var(--color-input-border) / <alpha-value>)",
          "card-bg": "rgb(var(--color-card-bg) / <alpha-value>)",
          "card-border": "rgb(var(--color-card-border) / <alpha-value>)",
          "dropdown-bg": "rgb(var(--color-dropdown-bg) / <alpha-value>)",
          "code-bg": "rgb(var(--color-code-bg) / <alpha-value>)",
          "inset-bg": "rgb(var(--color-inset-bg) / <alpha-value>)",
          "tab-active-bg": "rgb(var(--color-tab-active-bg) / <alpha-value>)",
          "tab-active-fg": "rgb(var(--color-tab-active-fg) / <alpha-value>)",
        },
      },
      animation: {
        "fade-in": "fadeIn 0.25s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
        "pop-in": "popIn 0.14s cubic-bezier(0.16, 1, 0.3, 1)",
        "live-sweep": "liveSweep 1.8s ease-in-out infinite",
        "dash-flow": "dashFlow 0.6s linear infinite",
      },
      keyframes: {
        dashFlow: {
          to: { "stroke-dashoffset": "-10" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        popIn: {
          "0%": { opacity: "0", transform: "scale(0.96) translateY(-4px)" },
          "100%": { opacity: "1", transform: "scale(1) translateY(0)" },
        },
        liveSweep: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(420%)" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 15px rgba(59, 130, 246, 0.1)" },
          "50%": { boxShadow: "0 0 25px rgba(59, 130, 246, 0.2)" },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
