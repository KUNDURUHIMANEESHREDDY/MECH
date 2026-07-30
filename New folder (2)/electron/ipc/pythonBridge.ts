import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";

import type { DesktopLogger } from "../logging";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
};

type PythonBridgeOptions = {
  pythonCommand: string;
  scriptPath: string;
  dbPath: string;
  logger: DesktopLogger;
};

export class PythonBridge {
  private process: ChildProcessWithoutNullStreams | null = null;
  private readonly pending = new Map<number, PendingRequest>();
  private nextId = 1;

  constructor(private readonly options: PythonBridgeOptions) {}

  async request<T = unknown>(
    method: string,
    params: Record<string, unknown> = {}
  ): Promise<T> {
    this.ensureStarted();
    const child = this.process;
    if (!child || !child.stdin.writable) {
      throw new Error("Python service is not writable.");
    }

    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params });

    return new Promise<T>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Python request timed out: ${method}`));
      }, 15_000);

      this.pending.set(id, {
        resolve: (value: unknown) => resolve(value as T),
        reject,
        timeout
      });

      child.stdin.write(`${payload}\n`, (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  stop(): void {
    if (!this.process) {
      return;
    }

    for (const [id, request] of this.pending) {
      clearTimeout(request.timeout);
      request.reject(new Error(`Python service stopped before request ${id} completed.`));
    }
    this.pending.clear();
    this.process.kill();
    this.process = null;
  }

  private ensureStarted(): void {
    if (this.process) {
      return;
    }

    const child = spawn(
      this.options.pythonCommand,
      ["-u", this.options.scriptPath, "--db", this.options.dbPath],
      {
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true
      }
    );
    this.process = child;
    this.options.logger.info("Python service started", {
      scriptPath: this.options.scriptPath,
      dbPath: this.options.dbPath
    });

    const stdout = createInterface({ input: child.stdout });
    stdout.on("line", (line) => this.handleLine(line));

    child.stderr.on("data", (chunk: Buffer) => {
      this.options.logger.warn("Python stderr", {
        message: chunk.toString("utf8").trim()
      });
    });

    child.on("exit", (code, signal) => {
      this.options.logger.info("Python service exited", { code, signal });
      this.process = null;
      for (const [id, request] of this.pending) {
        clearTimeout(request.timeout);
        request.reject(new Error(`Python service exited before request ${id} completed.`));
      }
      this.pending.clear();
    });
  }

  private handleLine(line: string): void {
    let message: {
      id?: number;
      result?: unknown;
      error?: { code?: string; message?: string };
    };
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.options.logger.error("Invalid Python JSON response", {
        line,
        error: String(error)
      });
      return;
    }

    if (typeof message.id !== "number") {
      this.options.logger.warn("Python response without request id", { line });
      return;
    }

    const request = this.pending.get(message.id);
    if (!request) {
      this.options.logger.warn("Python response for unknown request", {
        id: message.id
      });
      return;
    }

    clearTimeout(request.timeout);
    this.pending.delete(message.id);

    if (message.error) {
      request.reject(
        new Error(`${message.error.code ?? "PythonError"}: ${message.error.message ?? ""}`)
      );
      return;
    }

    request.resolve(message.result);
  }
}
