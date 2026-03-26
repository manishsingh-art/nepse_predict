/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#020617', // slate-950
        card: '#0f172a', // slate-900
        primary: '#38bdf8', // sky-400
        secondary: '#94a3b8', // slate-400
        bull: '#22c55e', // green-500
        bear: '#ef4444', // red-500
        neutral: '#eab308', // yellow-500
      }
    },
  },
  plugins: [],
}
