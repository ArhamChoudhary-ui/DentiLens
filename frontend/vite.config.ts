import { defineConfig } from 'vite'
import tailwind from '@tailwindcss/vite'

export default defineConfig({
  plugins: [tailwind()],
  build: { rollupOptions: { output: { manualChunks: { charts: ['recharts'] } } } },
  server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8502' } },
})
