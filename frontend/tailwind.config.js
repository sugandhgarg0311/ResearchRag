/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "rgb(var(--c-paper) / <alpha-value>)",
        panel: "rgb(var(--c-panel) / <alpha-value>)",
        card: "rgb(var(--c-card) / <alpha-value>)",
        ink: {
          DEFAULT: "rgb(var(--c-ink) / <alpha-value>)",
          soft: "rgb(var(--c-ink-soft) / <alpha-value>)",
          mute: "rgb(var(--c-ink-mute) / <alpha-value>)",
          faint: "rgb(var(--c-ink-faint) / <alpha-value>)",
        },
        line: "rgb(var(--c-line) / <alpha-value>)",
        accent: {
          DEFAULT: "rgb(var(--c-accent) / <alpha-value>)",
          dark: "rgb(var(--c-accent-dark) / <alpha-value>)",
          soft: "rgb(var(--c-accent-soft) / <alpha-value>)",
        },
        warn: {
          DEFAULT: "rgb(var(--c-warn) / <alpha-value>)",
          soft: "rgb(var(--c-warn-soft) / <alpha-value>)",
        },
        danger: {
          DEFAULT: "rgb(var(--c-danger) / <alpha-value>)",
          soft: "rgb(var(--c-danger-soft) / <alpha-value>)",
        },
        ok: {
          DEFAULT: "rgb(var(--c-ok) / <alpha-value>)",
          soft: "rgb(var(--c-ok-soft) / <alpha-value>)",
        },
        infer: {
          DEFAULT: "rgb(var(--c-infer) / <alpha-value>)",
          soft: "rgb(var(--c-infer-soft) / <alpha-value>)",
        },
        external: {
          DEFAULT: "rgb(var(--c-external) / <alpha-value>)",
          soft: "rgb(var(--c-external-soft) / <alpha-value>)",
        },
      },
      fontFamily: {
        display: ['Georgia', '"Iowan Old Style"', '"Times New Roman"', "serif"],
        sans: [
          '-apple-system', "BlinkMacSystemFont", '"Segoe UI"', "Roboto",
          '"Helvetica Neue"', "Arial", "sans-serif",
        ],
        mono: ['"SFMono-Regular"', "Menlo", "Consolas", '"Liberation Mono"', "monospace"],
        math: [
          '"STIX Two Math"', '"XITS Math"', '"Latin Modern Math"', '"Cambria Math"',
          "Georgia", '"Times New Roman"', "serif",
        ],
      },
      maxWidth: { content: "72rem" },
      boxShadow: {
        card: "0 1px 2px rgba(28,25,23,0.04), 0 1px 1px rgba(28,25,23,0.03)",
        lift: "0 4px 16px rgba(28,25,23,0.08), 0 1px 3px rgba(28,25,23,0.05)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadein: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        shimmer: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        dotpulse: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.45", transform: "scale(0.8)" },
        },
      },
      animation: {
        rise: "rise 220ms ease-out both",
        fadein: "fadein 260ms ease-out both",
        shimmer: "shimmer 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
