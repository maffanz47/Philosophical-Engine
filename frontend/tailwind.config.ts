import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: '#F5E6C8',
        ink: '#1A1209',
        'deep-brown': '#2C1810',
        'aged-gold': '#C9A84C',
        'faded-gold': '#8B6914',
        'candlelight': '#E8C547',
        'dusty-sage': '#7A9E7E',
        'faded-red': '#8B3A3A',
        'muted-slate': '#4A5568',
        vellum: '#EDE0C4'
      },
      fontFamily: {
        'cinzel': ['Cinzel', 'serif'],
        'eb-garamond': ['EB Garamond', 'serif'],
        'jetbrains': ['JetBrains Mono', 'monospace']
      }
    },
  },
  plugins: [],
}

export default config