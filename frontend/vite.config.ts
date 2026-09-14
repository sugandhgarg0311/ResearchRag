import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api -> the FastAPI server on :8000 so the UI can use
// relative URLs in both dev and production (served by the API itself).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
