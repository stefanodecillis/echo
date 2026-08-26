import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * The frontend test runner.
 *
 * Separate from `vite.config.ts` on purpose: that file is the app's build, and
 * everything here is about a run that never builds the app. What the two share
 * is the one thing they have to — the `@/` alias, which is how every module
 * under `src` refers to its neighbours.
 *
 * No DOM environment. Everything under test is a pure function: what folds new
 * segments into the ones on screen, which job a screen should name, what a line
 * is labelled in a copied transcript, whether a name is one Echo made up. They
 * are the pieces whose docblocks record real incidents, and none of them has
 * ever needed a document to be wrong.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
