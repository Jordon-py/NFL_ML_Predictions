import react from '@vitejs/plugin-react-swc';
import { defineConfig } from 'vite';

// React Router v7 ships ESM files with a `"use client"` directive. Rollup warns
// about module-level directives during bundling even though this is safe for our
// client-only Vite app. Filter only that specific warning to keep `vite build`
// output clean while still surfacing real issues.
export default defineConfig({
  plugins: [react()],
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

