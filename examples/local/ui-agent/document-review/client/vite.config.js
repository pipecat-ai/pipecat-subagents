import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    dedupe: ["@pipecat-ai/client-js"],
  },
  server: {
    port: 5173,
  },
});