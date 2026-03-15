/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: '#ec5b13',
        brandDark: '#1a120e',
        brandSurface: '#2d1e16',
        brandBorder: 'rgba(236, 91, 19, 0.15)',
        "landing-primary": "#ec5b13",
        "background-light": "#f8f6f6",
        "background-dark": "#1a120e",
        "accent-blue": "#3b82f6"
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ["Public Sans", "sans-serif"]
      },
      borderRadius: {
        'custom': '8px',
      }
    }
  },
  plugins: [],
}
