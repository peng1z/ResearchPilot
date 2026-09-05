import { defineConfig } from "vitest/config";

export default defineConfig({
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
