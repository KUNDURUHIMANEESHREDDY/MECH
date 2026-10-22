import { defineConfig } from 'vitest/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vue from '@vitejs/plugin-vue';

const here = path.dirname(fileURLToPath(import.meta.url));

// `@vitejs/plugin-vue` compiles the shared UI primitives in `src/components/ui/`.
// The plugin is ESM-only, so this config is an `.mts` module: Vite loads it as
// ESM directly instead of bundling it to CommonJS, where the import would be
// rewritten into a `require` of an ESM file and fail.
//
// This file was `vitest.config.js`. It moved to `.mts` for that reason only;
// the alias, the jsdom environment, the setup file and the include pattern
// are unchanged.
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(here, 'src')
    }
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/vitest/setup.js'],
    include: ['tests/vitest/**/*.test.{js,jsx}']
  }
});
