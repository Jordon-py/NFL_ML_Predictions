import { defineConfig } from 'vite';
// Switch to SWC-based React plugin to avoid Babel helper issues
import react from '@vitejs/plugin-react-swc';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    esbuildOptions: {
      // Treat .js as JSX in src when needed
      loader: { '.js': 'jsx' }
    }
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/schedule': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/predict': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  },
  build: {
    outDir: 'dist',     // ✅ dist relative to frontend/
    sourcemap: true
  },
  resolve: {
    alias: { '@': '/src' }
  }
});
