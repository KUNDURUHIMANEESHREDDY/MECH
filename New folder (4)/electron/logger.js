'use strict';

const MAX_ENTRIES = 2000;

class Logger {
  constructor() {
    this.entries = [];
    this.storage = null;
  }

  attachStorage(storage) {
    this.storage = storage;
  }

  _push(level, event, meta) {
    const entry = { ts: new Date().toISOString(), level, event, meta: meta || null };
    this.entries.push(entry);
    if (this.entries.length > MAX_ENTRIES) this.entries.shift();
    if (this.storage) {
      try {
        this.storage.appendLog(level, event, meta || null);
      } catch {
        // never let logging crash the app
      }
    }
    // eslint-disable-next-line no-console
    const line = `[${entry.ts}] ${level.toUpperCase()} ${event}` +
      (meta ? ' ' + JSON.stringify(meta) : '');
    if (level === 'error') console.error(line);
    else if (level === 'warn') console.warn(line);
    else console.log(line);
    return entry;
  }

  info(event, meta) { return this._push('info', event, meta); }
  warn(event, meta) { return this._push('warn', event, meta); }
  error(event, meta) { return this._push('error', event, meta); }

  getEntries() {
    return this.entries.slice();
  }
}

let singleton = null;
function getLogger() {
  if (!singleton) singleton = new Logger();
  return singleton;
}

module.exports = { Logger, getLogger };
