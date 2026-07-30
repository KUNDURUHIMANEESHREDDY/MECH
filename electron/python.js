'use strict';

const { spawn } = require('node:child_process');
const path = require('node:path');
const readline = require('node:readline');

/**
 * PythonBridge spawns the Python sidecar and communicates via JSON-lines
 * over stdio. Each request is assigned an id; the matching response is
 * resolved when a line with that id arrives on stdout.
 *
 * The Python side implements the same protocol in `python/main.py`.
 */
class PythonBridge {
  constructor({ logger, pythonPath, scriptPath }) {
    this.logger = logger;
    this.pythonPath = pythonPath || process.env.PYTHON_PATH || (process.platform === 'win32' ? 'py' : 'python3');
    this.scriptPath = scriptPath || path.join(__dirname, '..', 'backend', 'main.py');
    this.proc = null;
    this.pid = null;
    this.pending = new Map();
    this.nextId = 1;
    this.rl = null;
  }

  start() {
    return new Promise((resolve, reject) => {
      try {
        this.proc = spawn(this.pythonPath, [this.scriptPath], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONPATH: path.join(__dirname, '..') }
        });
      } catch (err) {
        return reject(err);
      }

      this.pid = this.proc.pid || null;
      this.rl = readline.createInterface({ input: this.proc.stdout });

      this.rl.on('line', (line) => this._onLine(line));

      this.proc.stderr.on('data', (chunk) => {
        const text = chunk.toString();
        for (const line of text.split(/\r?\n/)) {
          if (line.trim()) this.logger.warn('python_stderr', { line });
        }
      });

      this.proc.on('error', (err) => {
        this.logger.error('python_process_error', { error: err.message });
        this._failAll(err);
      });

      this.proc.on('close', (code) => {
        this.logger.info('python_process_closed', { code });
        this._failAll(new Error(`python exited with code ${code}`));
        this.proc = null;
        this.pid = null;
      });

      // Wait briefly for the sidecar to signal it's ready.
      const onReady = (line) => {
        try {
          const msg = JSON.parse(line);
          if (msg && msg.type === 'ready') {
            this.rl.removeListener('line', onReady);
            resolve();
          }
        } catch {
          // ignore non-JSON noise during startup
        }
      };
      this.rl.on('line', onReady);

      // Hard timeout in case the sidecar never reports ready.
      // 15s to allow the large dispatcher module graph to finish importing.
      setTimeout(() => {
        this.rl.removeListener('line', onReady);
        resolve(); // proceed optimistically
      }, 15000);
    });
  }

  _onLine(line) {
    if (!line.trim()) return;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      this.logger.warn('python_non_json', { line });
      return;
    }
    if (msg && typeof msg.id === 'number' && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error));
      else resolve(msg.result);
    }
  }

  _failAll(err) {
    for (const [, { reject }] of this.pending) reject(err);
    this.pending.clear();
  }

  call(method, payload) {
    if (!this.proc) {
      return Promise.reject(new Error('python_bridge_not_started'));
    }
    const id = this.nextId++;
    const message = JSON.stringify({ id, method, payload: payload || {} });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(message + '\n', (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  async stop() {
    if (!this.proc) return;
    try {
      this.proc.stdin.end();
    } catch {
      // ignore
    }
    try {
      this.proc.kill();
    } catch {
      // ignore
    }
    this.proc = null;
    this.pid = null;
  }
}

let singleton = null;
function getPythonBridge(opts) {
  if (!singleton) singleton = new PythonBridge(opts || {});
  return singleton;
}

module.exports = { PythonBridge, getPythonBridge };
