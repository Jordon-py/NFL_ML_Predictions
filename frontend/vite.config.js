import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.[jt]sx?$/,
    exclude: [],
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: {
        '.js': 'jsx',
      },
    },
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com',
        changeOrigin: true,
        secure: true,
      },
      // Proxy schedule endpoint
      '/schedule': {
        target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com',
        changeOrigin: true,
        secure: true,
      },
      // Proxy predict endpoint
      '/predict': {
        target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com',
        changeOrigin: true,
        secure: true,
      },
    },
  },
  build: {
    outDir: 'build',
    sourcemap: true,
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
});
