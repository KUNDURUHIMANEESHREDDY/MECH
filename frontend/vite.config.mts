import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import tailwindcss from '@tailwindcss/vite';

/**
 * Bearer token for the dev proxy only.
 *
 * The renderer never holds the token: in the desktop build the preload's
 * `api:request` proxy signs requests in the main process. A plain browser has
 * no such process, so `src/services/api.ts` sends no Authorization header and
 * every /api call answered 401 -- which left `npm run dev:renderer` and the
 * Playwright live specs unable to see live data at all.
 *
 * Reading the token here (dev-server process, from the environment or the
 * backend's own token file) keeps it out of the bundle: nothing under src/
 * knows it, and a production `vite build` never evaluates this proxy.
 */
function devBearerToken(): string | null {
  const fromEnv = process.env.MECH_API_TOKEN?.trim();
  if (fromEnv) return fromEnv;
  try {
    const token = readFileSync(
      resolve(__dirname, '../backend/storage/.mech_api_token'),
      'utf8',
    ).trim();
    return token || null;
  } catch {
    return null;
  }
}

const token = devBearerToken();

// `react()` and `vueJsx()` were registered here for six React files that nothing
// imported. Those files are gone, so both plugins and their packages are removed.
// Keeping them cost a JSX transform on every module for no .jsx or .tsx source.
export default defineConfig({
  base: './',
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
      },
    },
    // The dev server watched the whole frontend tree, including build/python/**
    // (a bundled CPython with thousands of site-package files). A pip install or
    // app build writing there triggered one full page reload per file -- 648 in
    // a minute -- which reset the active route mid-navigation and made every
    // click look broken. Nothing under these paths is app source.
    watch: {
      ignored: [
        // Editing a spec or a harness reloaded the running app: the page was
        // torn down mid-run, which looked like a product failure (a destroyed
        // execution context, a panel that reset itself). Tests are not app
        // source and nothing in src/ imports them.
        '**/tests/**',
        '**/test-results/**',
        '**/build/**',
        '**/release/**',
        '**/dist/**',
        '**/.venv/**',
        '**/test-results/**',
        '**/screenshots/**',
        '**/*.log',
      ],
    },
  },
});
