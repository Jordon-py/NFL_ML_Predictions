// ==========================================
// File: frontend/vite.config.js
// Role: Vite build configuration.
// Input Data: Build environment variables.
// Output Data: Vite config object.
// Dependencies: @vitejs/plugin-react-swc, vite
// Notes: Used by Vite CLI.
// ==========================================

import react from '@vitejs/plugin-react-swc';
import { defineConfig } from 'vite';

// React Router v7 ships ESM files with a `"use client"` directive. Rollup warns
// about module-level directives during bundling even though this is safe for our
// client-only Vite app. Filter only that specific warning to keep `vite build`
// output clean while still surfacing real issues.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Local dev: avoid CORS by proxying API calls to FastAPI.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        if (
          warning.code === 'MODULE_LEVEL_DIRECTIVE' &&
          typeof warning.message === 'string' &&
          warning.message.includes('"use client"')
        ) {
          return;
        }
        warn(warning);
      },
    },
  },
});

