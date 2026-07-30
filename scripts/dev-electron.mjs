import { spawn } from "node:child_process";

import electronPath from "electron";
import { createServer } from "vite";

const server = await createServer({
  configFile: "vite.config.ts",
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});

await server.listen();
const address = server.httpServer?.address();
const port = typeof address === "object" && address ? address.port : 5173;
const url = `http://127.0.0.1:${port}`;

const child = spawn(electronPath, ["."], {
  stdio: "inherit",
  env: {
    ...process.env,
    VITE_DEV_SERVER_URL: url
  }
});

child.on("exit", async (code) => {
  await server.close();
  process.exit(code ?? 0);
});

process.on("SIGINT", () => {
  child.kill();
});
