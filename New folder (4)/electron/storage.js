'use strict';

const path = require('node:path');
const fs = require('node:fs');
const Database = require('better-sqlite3');

const DEFAULT_DIR = path.join(__dirname, '..', 'storage');
const DEFAULT_DB = path.join(DEFAULT_DIR, 'app.db');

class SettingsStore {
  constructor(db) {
    this.db = db;
  }

  get(key) {
    const row = this.db.prepare('SELECT value FROM kv WHERE key = ?').get(key);
    if (!row) return null;
    try {
      return JSON.parse(row.value);
    } catch {
      return null;
    }
  }

  set(key, value) {
    const json = JSON.stringify(value);
    this.db
      .prepare('INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value')
      .run(key, json);
  }
}

class ProjectStore {
  constructor(db) {
    this.db = db;
  }

  list() {
    return this.db.prepare('SELECT * FROM projects ORDER BY rowid DESC').all();
  }

  upsert(project) {
    const sql = `INSERT INTO projects (id, name, path, createdAt, updatedAt) VALUES (?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET name = excluded.name, path = excluded.path, updatedAt = excluded.updatedAt`;
    this.db.prepare(sql).run(project.id, project.name, project.path || '', project.createdAt, project.updatedAt);
    return project;
  }

  remove(id) {
    const res = this.db.prepare('DELETE FROM projects WHERE id = ?').run(id);
    return res.changes > 0;
  }
}

class SessionStore {
  constructor(db) {
    this.db = db;
  }

  list() {
    return this.db.prepare('SELECT * FROM sessions ORDER BY updatedAt DESC').all().map((r) => ({
      ...r,
      metadata: r.metadata ? safeParse(r.metadata) : {}
    }));
  }

  upsert(session) {
    const sql = `INSERT INTO sessions (id, name, model, prompt, createdAt, updatedAt, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET name = excluded.name, model = excluded.model, prompt = excluded.prompt, updatedAt = excluded.updatedAt, metadata = excluded.metadata`;
    this.db.prepare(sql).run(
      session.id,
      session.name || 'Untitled Session',
      session.model || 'GPT-2',
      session.prompt || '',
      session.createdAt || new Date().toISOString(),
      new Date().toISOString(),
      session.metadata ? JSON.stringify(session.metadata) : null
    );
    return session;
  }

  remove(id) {
    const res = this.db.prepare('DELETE FROM sessions WHERE id = ?').run(id);
    return res.changes > 0;
  }
}

class ExperimentStore {
  constructor(db) {
    this.db = db;
  }

  list() {
    return this.db.prepare('SELECT * FROM experiments ORDER BY updatedAt DESC').all().map((r) => ({
      ...r,
      results: r.results ? safeParse(r.results) : {}
    }));
  }

  upsert(exp) {
    const sql = `INSERT INTO experiments (id, title, targetCircuit, status, createdAt, updatedAt, results) VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET title = excluded.title, targetCircuit = excluded.targetCircuit, status = excluded.status, updatedAt = excluded.updatedAt, results = excluded.results`;
    this.db.prepare(sql).run(
      exp.id,
      exp.title || 'Untitled Experiment',
      exp.targetCircuit || 'L0_N0',
      exp.status || 'pending',
      exp.createdAt || new Date().toISOString(),
      new Date().toISOString(),
      exp.results ? JSON.stringify(exp.results) : null
    );
    return exp;
  }

  remove(id) {
    const res = this.db.prepare('DELETE FROM experiments WHERE id = ?').run(id);
    return res.changes > 0;
  }
}

class StorageManager {
  constructor(opts) {
    this.dir = (opts && opts.dir) || DEFAULT_DIR;
    this.dbPath = (opts && opts.dbPath) || DEFAULT_DB;
    this.db = null;
    this.settings = null;
    this.projects = null;
    this.sessions = null;
    this.experiments = null;
  }

  async init() {
    if (!fs.existsSync(this.dir)) {
      fs.mkdirSync(this.dir, { recursive: true });
    }
    this.db = new Database(this.dbPath);
    this.db.pragma('journal_mode = WAL');
    this._migrate();

    this.settings = new SettingsStore(this.db);
    this.projects = new ProjectStore(this.db);
    this.sessions = new SessionStore(this.db);
    this.experiments = new ExperimentStore(this.db);
    return this;
  }

  _migrate() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        path TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS recent_files (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        label TEXT,
        openedAt TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model TEXT,
        prompt TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL,
        metadata TEXT
      );

      CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        targetCircuit TEXT,
        status TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL,
        results TEXT
      );

      CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        event TEXT NOT NULL,
        meta TEXT
      );
    `);
  }

  // Backward compatible KV methods
  async get(key) {
    return this.settings.get(key);
  }

  async set(key, value) {
    return this.settings.set(key, value);
  }

  // Backward compatible table methods
  async list(table) {
    if (table === 'projects') return this.projects.list();
    if (table === 'sessions') return this.sessions.list();
    if (table === 'experiments') return this.experiments.list();
    if (table === 'recent_files') {
      return this.db.prepare('SELECT * FROM recent_files ORDER BY rowid DESC').all();
    }
    throw new Error(`invalid table: ${table}`);
  }

  async upsert(table, record) {
    if (table === 'projects') return this.projects.upsert(record);
    if (table === 'sessions') return this.sessions.upsert(record);
    if (table === 'experiments') return this.experiments.upsert(record);
    if (table === 'recent_files') {
      const sql = `INSERT INTO recent_files (id, path, label, openedAt) VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET path = excluded.path, label = excluded.label, openedAt = excluded.openedAt`;
      this.db.prepare(sql).run(record.id, record.path, record.label || '', record.openedAt);
      return record;
    }
    throw new Error(`invalid table: ${table}`);
  }

  async remove(table, where) {
    const id = where && where.id;
    if (!id) return false;
    if (table === 'projects') return this.projects.remove(id);
    if (table === 'sessions') return this.sessions.remove(id);
    if (table === 'experiments') return this.experiments.remove(id);
    if (table === 'recent_files') {
      const res = this.db.prepare('DELETE FROM recent_files WHERE id = ?').run(id);
      return res.changes > 0;
    }
    return false;
  }

  async clear(table) {
    if (['projects', 'sessions', 'experiments', 'recent_files', 'logs'].includes(table)) {
      this.db.prepare(`DELETE FROM ${table}`).run();
    }
  }

  appendLog(level, event, meta) {
    this.db
      .prepare('INSERT INTO logs(ts, level, event, meta) VALUES(?, ?, ?, ?)')
      .run(new Date().toISOString(), level, event, meta ? JSON.stringify(meta) : null);
  }

  getEntries(limit = 500) {
    return this.db
      .prepare('SELECT id, ts, level, event, meta FROM logs ORDER BY id DESC LIMIT ?')
      .all(limit)
      .map((r) => ({ ...r, meta: r.meta ? safeParse(r.meta) : null }))
      .reverse();
  }

  close() {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

let singleton = null;
function getStorage(opts) {
  if (!singleton) singleton = new StorageManager(opts || {});
  return singleton;
}

module.exports = {
  StorageManager,
  Storage: StorageManager,
  getStorage,
  SettingsStore,
  ProjectStore,
  SessionStore,
  ExperimentStore
};
