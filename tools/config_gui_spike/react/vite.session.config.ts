import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repository = resolve(here, "../../..");

export default defineConfig({
  root: repository,
  plugins: [react()],
  server: {
    fs: { allow: [repository] },
  },
  resolve: {
    alias: [
      {
        find: /^react\/jsx-dev-runtime$/,
        replacement: resolve(here, "node_modules/react/jsx-dev-runtime.js"),
      },
      {
        find: /^react\/jsx-runtime$/,
        replacement: resolve(here, "node_modules/react/jsx-runtime.js"),
      },
      {
        find: /^react$/,
        replacement: resolve(here, "node_modules/react/index.js"),
      },
      {
        find: /^react-dom\/(.*)$/,
        replacement: resolve(here, "node_modules/react-dom/$1"),
      },
      {
        find: /^react-dom$/,
        replacement: resolve(here, "node_modules/react-dom/index.js"),
      },
      {
        find: /^@testing-library\/(.*)$/,
        replacement: resolve(here, "node_modules/@testing-library/$1"),
      },
    ],
  },
  test: {
    environment: "jsdom",
    setupFiles: resolve(here, "src/test-setup.ts"),
    include: ["tests/gui/react/**/*.test.tsx"],
  },
  build: {
    lib: {
      entry: resolve(repository, "src/rheplicant/gui/react/SessionEditor.tsx"),
      formats: ["es"],
      fileName: "session-editor",
    },
    outDir: resolve(here, "dist/session"),
    emptyOutDir: true,
    rollupOptions: {
      external: ["react", "react-dom", "react/jsx-runtime"],
    },
  },
});
