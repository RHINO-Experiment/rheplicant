import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repository = resolve(here, "../../..");
const source = resolve(repository, "src/rheplicant/gui/react");

export default defineConfig({
  root: source,
  plugins: [react()],
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
    ],
  },
  build: {
    outDir: resolve(repository, "src/rheplicant/gui/static"),
    emptyOutDir: true,
    sourcemap: false,
  },
});
