import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

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
      '/api': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true, secure: true },
      '/schedule': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true, secure: true },
      '/predict': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true, secure: true }
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
