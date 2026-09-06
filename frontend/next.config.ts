import type { NextConfig } from "next";

// Two deployment shapes share this app.
//
// `standalone` is what docker-compose builds: the frontend Dockerfile runs
// `npm run start`, which boots .next/standalone/server.js.
//
// `export` emits plain static files for the hosted demo. Every page here is a
// client component that talks to an API base chosen at runtime, so there is
// nothing for a Node server to do -- but the two outputs are mutually
// exclusive, so the static build opts in through `npm run build:static`
// rather than the mode being switched globally.
const nextConfig: NextConfig = {
  // Directory-style output, so a case URL survives a refresh on any static
  // host. Without it the export writes runs/<slug>.html and a plain server
  // answers 404 for /runs/<slug> -- verified, not assumed.
  trailingSlash: true,
  output: process.env.NEXT_OUTPUT === "export" ? "export" : "standalone",
};

export default nextConfig;
