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
  output: process.env.NEXT_OUTPUT === "export" ? "export" : "standalone",
};

export default nextConfig;
