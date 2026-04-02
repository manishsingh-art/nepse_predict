import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Trading dashboard dark palette
        surface: {
          DEFAULT: "#0f1629",
          secondary: "#0a0e1a",
          card: "#111827",
          border: "#1e2a45",
          hover: "#162032",
        },
        action: {
          buy: "#10b981",
          "buy-dim": "#064e3b",
          sell: "#ef4444",
          "sell-dim": "#7f1d1d",
          hold: "#f59e0b",
          "hold-dim": "#78350f",
          avoid: "#8b5cf6",
          "avoid-dim": "#4c1d95",
        },
        chart: {
          primary: "#3b82f6",
          forecast: "#f59e0b",
          actual: "#10b981",
          grid: "#1e2a45",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        shimmer: "shimmer 2s linear infinite",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
