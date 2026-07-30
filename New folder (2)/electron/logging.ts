import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

import type { LogEntry } from "./types";

type LogLevel = LogEntry["level"];
type Broadcaster = (entry: LogEntry) => void;

export class DesktopLogger {
  private readonly entries: LogEntry[] = [];
  private broadcaster: Broadcaster | null = null;
  readonly logPath: string;

  constructor(userDataPath: string) {
    const logDir = path.join(userDataPath, "logs");
    mkdirSync(logDir, { recursive: true });
    this.logPath = path.join(logDir, "desktop.log");
  }

  setBroadcaster(broadcaster: Broadcaster): void {
    this.broadcaster = broadcaster;
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.write("info", message, context);
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.write("warn", message, context);
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.write("error", message, context);
  }

  getEntries(): LogEntry[] {
    if (this.entries.length > 0) {
      return [...this.entries];
    }

    try {
      return readFileSync(this.logPath, "utf8")
        .split(/\r?\n/)
        .filter(Boolean)
        .slice(-200)
        .map((line) => JSON.parse(line) as LogEntry);
    } catch {
      return [];
    }
  }

  private write(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>
  ): void {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context
    };
    this.entries.push(entry);
    while (this.entries.length > 200) {
      this.entries.shift();
    }

    appendFileSync(this.logPath, `${JSON.stringify(entry)}\n`, "utf8");
    this.broadcaster?.(entry);
  }
}
