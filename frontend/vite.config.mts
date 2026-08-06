import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import react from '@vitejs/plugin-react';
import vueJsx from '@vitejs/plugin-vue-jsx';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  base: './',
  plugins: [vue(), react(), vueJsx(), tailwindcss()],
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
      },
    },
  },
});
