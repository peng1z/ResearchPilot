import { defineConfig } from "vitest/config";

export default defineConfig({
  // Next compiles JSX with the automatic runtime, so a server component has
  // no reason to import React. Without this, esbuild falls back to the
  // classic transform and those files fail at import with "React is not
  // defined" -- a difference between the test environment and the real one,
  // not a defect in the page.
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    // The deployed demo is served over https, and some behaviour depends on
    // it -- a plain http backend is blocked as mixed content there. jsdom
    // defaults to http, which would hide that.
    environmentOptions: { jsdom: { url: "https://researchpilot.example/" } },
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
  },
});
