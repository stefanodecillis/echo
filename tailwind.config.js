/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Sana-style palette: white, near-black text, one accent used sparingly.
      colors: {
        ink: {
          DEFAULT: "#111113",
          soft: "#3f3f46",
          faint: "#71717a",
          ghost: "#a1a1aa",
        },
        hairline: "#eeeeee",
        surface: {
          DEFAULT: "#ffffff",
          sunken: "#fafafa",
        },
        accent: {
          DEFAULT: "#4f46e5",
          soft: "#eef2ff",
        },
        live: "#e11d48",
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      borderRadius: {
        xl: "0.875rem",
        "2xl": "1.25rem",
      },
      boxShadow: {
        card: "0 1px 2px rgba(17, 17, 19, 0.04)",
        lift: "0 8px 24px rgba(17, 17, 19, 0.08)",
      },
    },
  },
  plugins: [],
};
