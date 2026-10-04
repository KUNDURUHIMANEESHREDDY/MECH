import { defineConfig } from 'vitest/config';
import path from 'node:path';

// Was `plugins: [react()]`, for the same six removed React files.
//
// No plugin is registered now. The suite's three tests import `src/services/api`
// and a Pinia store -- plain JavaScript modules, no `.vue` component and no JSX
// -- so there is nothing for a component plugin to transform. `@vitejs/plugin-vue`
// was tried here and cannot be: it is ESM-only, and this file is CommonJS
// (`"type": "commonjs"`), so `require` of it fails outright. Adding a plugin that
// transforms nothing would only add a way for the config to break.
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/vitest/setup.js'],
    include: ['tests/vitest/**/*.test.{js,jsx}']
  }
});
