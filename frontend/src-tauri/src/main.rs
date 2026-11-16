use mech_core::ApiProxyRequest;
use mech_runtime::BackendSupervisor;
use reqwest::{Client, Method};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};
use tauri::State;

struct AppStorage {
    db: Connection,
}

impl AppStorage {
    fn open(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create storage directory: {e}"))?;
        }
        let db = Connection::open(path).map_err(|e| format!("open MECH storage: {e}"))?;
        db.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| e.to_string())?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT, createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS recent_files (id TEXT PRIMARY KEY, path TEXT NOT NULL, label TEXT, openedAt TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT, prompt TEXT, createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL, metadata TEXT);
             CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, title TEXT NOT NULL, targetCircuit TEXT, status TEXT, createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL, results TEXT);
             CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, level TEXT NOT NULL, event TEXT NOT NULL, meta TEXT);"
        ).map_err(|e| format!("initialize MECH storage: {e}"))?;
        Ok(Self { db })
    }

    fn list(&self, table: &str) -> Result<Vec<Value>, String> {
        match table {
            "projects" => {
                let mut stmt = self
                    .db
                    .prepare(
                        "SELECT id,name,path,createdAt,updatedAt FROM projects ORDER BY rowid DESC",
                    )
                    .map_err(|e| e.to_string())?;
                let rows = stmt.query_map([], |r| Ok(json!({"id":r.get::<_,String>(0)?,"name":r.get::<_,String>(1)?,"path":r.get::<_,Option<String>>(2)?,"createdAt":r.get::<_,String>(3)?,"updatedAt":r.get::<_,String>(4)?}))).map_err(|e| e.to_string())?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())
            }
            "recent_files" => {
                let mut stmt = self
                    .db
                    .prepare("SELECT id,path,label,openedAt FROM recent_files ORDER BY rowid DESC")
                    .map_err(|e| e.to_string())?;
                let rows = stmt.query_map([], |r| Ok(json!({"id":r.get::<_,String>(0)?,"path":r.get::<_,String>(1)?,"label":r.get::<_,Option<String>>(2)?,"openedAt":r.get::<_,String>(3)?}))).map_err(|e| e.to_string())?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())
            }
            "sessions" => {
                let mut stmt = self.db.prepare("SELECT id,name,model,prompt,createdAt,updatedAt,metadata FROM sessions ORDER BY updatedAt DESC").map_err(|e| e.to_string())?;
                let rows = stmt.query_map([], |r| {
                    let metadata: Option<String> = r.get(6)?;
                    Ok(json!({"id":r.get::<_,String>(0)?,"name":r.get::<_,String>(1)?,"model":r.get::<_,Option<String>>(2)?,"prompt":r.get::<_,Option<String>>(3)?,"createdAt":r.get::<_,String>(4)?,"updatedAt":r.get::<_,String>(5)?,"metadata":metadata.and_then(|v| serde_json::from_str::<Value>(&v).ok()).unwrap_or_else(|| json!({}))}))
                }).map_err(|e| e.to_string())?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())
            }
            "experiments" => {
                let mut stmt = self.db.prepare("SELECT id,title,targetCircuit,status,createdAt,updatedAt,results FROM experiments ORDER BY updatedAt DESC").map_err(|e| e.to_string())?;
                let rows = stmt.query_map([], |r| {
                    let results: Option<String> = r.get(6)?;
                    Ok(json!({"id":r.get::<_,String>(0)?,"title":r.get::<_,String>(1)?,"targetCircuit":r.get::<_,Option<String>>(2)?,"status":r.get::<_,Option<String>>(3)?,"createdAt":r.get::<_,String>(4)?,"updatedAt":r.get::<_,String>(5)?,"results":results.and_then(|v| serde_json::from_str::<Value>(&v).ok()).unwrap_or_else(|| json!({}))}))
                }).map_err(|e| e.to_string())?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())
            }
            _ => Err(format!("invalid storage table: {table}")),
        }
    }

    fn upsert(&self, table: &str, record: &Value) -> Result<Value, String> {
        let id = record
            .get("id")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| "record requires a non-empty string id".to_owned())?;
        let now = now_iso();
        match table {
            "projects" => self.db.execute("INSERT INTO projects(id,name,path,createdAt,updatedAt) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET name=excluded.name,path=excluded.path,updatedAt=excluded.updatedAt", params![id, record.get("name").and_then(Value::as_str).unwrap_or("Untitled Workspace"), record.get("path").and_then(Value::as_str).unwrap_or(""), record.get("createdAt").and_then(Value::as_str).unwrap_or(&now), now]).map_err(|e| e.to_string())?,
            "recent_files" => self.db.execute("INSERT INTO recent_files(id,path,label,openedAt) VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET path=excluded.path,label=excluded.label,openedAt=excluded.openedAt", params![id, record.get("path").and_then(Value::as_str).ok_or("recent file requires path")?, record.get("label").and_then(Value::as_str).unwrap_or(""), record.get("openedAt").and_then(Value::as_str).unwrap_or(&now)]).map_err(|e| e.to_string())?,
            "sessions" => self.db.execute("INSERT INTO sessions(id,name,model,prompt,createdAt,updatedAt,metadata) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(id) DO UPDATE SET name=excluded.name,model=excluded.model,prompt=excluded.prompt,updatedAt=excluded.updatedAt,metadata=excluded.metadata", params![id, record.get("name").and_then(Value::as_str).unwrap_or("Untitled Session"), record.get("model").and_then(Value::as_str).unwrap_or("GPT-2"), record.get("prompt").and_then(Value::as_str).unwrap_or(""), record.get("createdAt").and_then(Value::as_str).unwrap_or(&now), now, record.get("metadata").map(Value::to_string)]).map_err(|e| e.to_string())?,
            "experiments" => self.db.execute("INSERT INTO experiments(id,title,targetCircuit,status,createdAt,updatedAt,results) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(id) DO UPDATE SET title=excluded.title,targetCircuit=excluded.targetCircuit,status=excluded.status,updatedAt=excluded.updatedAt,results=excluded.results", params![id, record.get("title").and_then(Value::as_str).unwrap_or("Untitled Experiment"), record.get("targetCircuit").and_then(Value::as_str).unwrap_or("L0_N0"), record.get("status").and_then(Value::as_str).unwrap_or("pending"), record.get("createdAt").and_then(Value::as_str).unwrap_or(&now), now, record.get("results").map(Value::to_string)]).map_err(|e| e.to_string())?,
            _ => return Err(format!("invalid storage table: {table}")),
        };
        Ok(record.clone())
    }

    fn remove(&self, table: &str, id: &str) -> Result<bool, String> {
        let sql = match table {
            "projects" => "DELETE FROM projects WHERE id=?1",
            "recent_files" => "DELETE FROM recent_files WHERE id=?1",
            "sessions" => "DELETE FROM sessions WHERE id=?1",
            "experiments" => "DELETE FROM experiments WHERE id=?1",
            _ => return Err(format!("invalid storage table: {table}")),
        };
        self.db
            .execute(sql, [id])
            .map(|n| Ok(n > 0))
            .map_err(|e| e.to_string())?
    }

    fn clear(&self, table: &str) -> Result<(), String> {
        let sql = match table {
            "projects" => "DELETE FROM projects",
            "recent_files" => "DELETE FROM recent_files",
            "sessions" => "DELETE FROM sessions",
            "experiments" => "DELETE FROM experiments",
            "logs" => "DELETE FROM logs",
            _ => return Err(format!("invalid storage table: {table}")),
        };
        self.db
            .execute(sql, [])
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn get_setting(&self) -> Result<Option<Value>, String> {
        let raw: Option<String> = self
            .db
            .query_row("SELECT value FROM kv WHERE key='settings'", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(raw.and_then(|v| serde_json::from_str(&v).ok()))
    }

    fn set_setting(&self, value: &Value) -> Result<(), String> {
        self.db.execute("INSERT INTO kv(key,value) VALUES('settings',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [value.to_string()]).map(|_| ()).map_err(|e| e.to_string())
    }
}

struct AppState {
    root: PathBuf,
    storage: Mutex<AppStorage>,
    client: Client,
    backend_status: Arc<Mutex<String>>,
    _backend: Arc<Mutex<Option<BackendSupervisor>>>,
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn default_settings() -> Value {
    let python = if cfg!(windows) { "python" } else { "python3" };
    json!({
        "theme":"system",
        "gpu":{"enabled":true,"acceleration":"auto"},
        "cache":{"enabled":true,"maxSizeMb":1024,"location":""},
        "paths":{"python":python,"workspace":"","projects":""},
        "models":{"defaultModel":"GPT-2 Small","device":"cpu","maxContextTokens":2048,"precision":"float32"},
        "python":{"path":python,"environment":"system","venvPath":""},
        "performance":{"threads":4,"memoryLimitGb":8,"enableTorchCompile":false},
        "plugins":{"enabledPlugins":["circuit-vis","token-heatmaps"],"autoUpdate":true},
        "debugger":{"breakpointBehavior":"pause","tokenHighlighting":"activation","traceLevel":"medium"},
        "logging":{"level":"info","fileRetentionDays":7,"consoleOutput":true}
    })
}

fn merge_json(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(dst), Value::Object(src)) => {
            for (key, value) in src {
                if let Some(existing) = dst.get_mut(key) {
                    merge_json(existing, value);
                } else {
                    dst.insert(key.clone(), value.clone());
                }
            }
        }
        (dst, src) => *dst = src.clone(),
    }
}

fn resolve_token(root: &Path) -> Result<String, String> {
    if let Ok(token) = std::env::var("MECH_API_TOKEN") {
        let token = token.trim().to_owned();
        if !token.is_empty() {
            return Ok(token);
        }
    }
    let path = root.join("backend/storage/.mech_api_token");
    std::fs::read_to_string(&path)
        .map(|v| v.trim().to_owned())
        .map_err(|_| format!("API token unavailable at {}", path.display()))
}

async fn send_api_request(state: &AppState, request: ApiProxyRequest) -> Result<Value, String> {
    let request = request.validate()?;
    let token = resolve_token(&state.root)?;
    let method = Method::from_bytes(request.method.as_bytes()).map_err(|e| e.to_string())?;
    let url = format!("http://127.0.0.1:8000{}", request.path);
    let mut builder = state
        .client
        .request(method, url)
        .timeout(Duration::from_secs(180))
        .bearer_auth(token);
    if !request.query.is_empty() {
        builder = builder.query(&request.query);
    }
    for (name, value) in request.headers {
        builder = builder.header(name, value);
    }
    if let Some(body) = request.body {
        builder = builder.json(&body);
    }
    let response = builder
        .send()
        .await
        .map_err(|e| format!("MECH API transport failed: {e}"))?;
    let status = response.status().as_u16();
    let text = response
        .text()
        .await
        .map_err(|e| format!("read MECH API response: {e}"))?;
    let body = if text.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&text).unwrap_or(Value::String(text))
    };
    Ok(json!({"status":status,"body":body}))
}

#[tauri::command]
async fn api_request(
    state: State<'_, AppState>,
    request: ApiProxyRequest,
) -> Result<Value, String> {
    send_api_request(&state, request).await
}

#[tauri::command]
async fn backend_call(
    state: State<'_, AppState>,
    method: String,
    payload: Option<Value>,
) -> Result<Value, String> {
    if method.is_empty()
        || !method
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '/')
    {
        return Err("invalid backend method".into());
    }
    let request = ApiProxyRequest {
        method: "POST".into(),
        path: format!("/api/{method}"),
        body: Some(payload.unwrap_or_else(|| json!({}))),
        query: None,
        headers: None,
    };
    let response = send_api_request(&state, request).await?;
    let status = response.get("status").and_then(Value::as_u64).unwrap_or(0);
    if !(200..300).contains(&status) {
        return Err(format!("backend {method} -> HTTP {status}"));
    }
    let body = response.get("body").cloned().unwrap_or(Value::Null);
    Ok(body.get("result").cloned().unwrap_or(body))
}

#[tauri::command]
fn backend_status(state: State<'_, AppState>) -> String {
    state
        .backend_status
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| "ERROR: status lock poisoned".into())
}

#[tauri::command]
fn settings_get(state: State<'_, AppState>) -> Result<Value, String> {
    let storage = state.storage.lock().map_err(|_| "storage lock poisoned")?;
    let mut settings = default_settings();
    if let Some(saved) = storage.get_setting()? {
        merge_json(&mut settings, &saved);
    }
    Ok(settings)
}

#[tauri::command]
fn settings_set(state: State<'_, AppState>, patch: Value) -> Result<Value, String> {
    let storage = state.storage.lock().map_err(|_| "storage lock poisoned")?;
    let mut settings = storage.get_setting()?.unwrap_or_else(default_settings);
    merge_json(&mut settings, &patch);
    storage.set_setting(&settings)?;
    Ok(settings)
}

#[tauri::command]
fn settings_reset(state: State<'_, AppState>) -> Result<Value, String> {
    let settings = default_settings();
    state
        .storage
        .lock()
        .map_err(|_| "storage lock poisoned")?
        .set_setting(&settings)?;
    Ok(settings)
}

#[tauri::command]
fn storage_list(state: State<'_, AppState>, table: String) -> Result<Vec<Value>, String> {
    state
        .storage
        .lock()
        .map_err(|_| "storage lock poisoned")?
        .list(&table)
}

#[tauri::command]
fn storage_upsert(
    state: State<'_, AppState>,
    table: String,
    record: Value,
) -> Result<Value, String> {
    state
        .storage
        .lock()
        .map_err(|_| "storage lock poisoned")?
        .upsert(&table, &record)
}

#[tauri::command]
fn storage_remove(state: State<'_, AppState>, table: String, id: String) -> Result<bool, String> {
    state
        .storage
        .lock()
        .map_err(|_| "storage lock poisoned")?
        .remove(&table, &id)
}

#[tauri::command]
fn storage_clear(state: State<'_, AppState>, table: String) -> Result<(), String> {
    state
        .storage
        .lock()
        .map_err(|_| "storage lock poisoned")?
        .clear(&table)
}

#[tauri::command]
fn storage_logs(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    let storage = state.storage.lock().map_err(|_| "storage lock poisoned")?;
    let mut stmt = storage
        .db
        .prepare("SELECT id,ts,level,event,meta FROM logs ORDER BY id DESC LIMIT 500")
        .map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| {
        let meta: Option<String> = r.get(4)?;
        Ok(json!({"id":r.get::<_,i64>(0)?,"timestamp":r.get::<_,String>(1)?,"level":r.get::<_,String>(2)?,"message":r.get::<_,String>(3)?,"context":meta.and_then(|v| serde_json::from_str::<Value>(&v).ok())}))
    }).map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

fn find_repo_root() -> Result<PathBuf, String> {
    if let Ok(root) = std::env::var("MECH_REPO_ROOT") {
        let path = PathBuf::from(root);
        if path.join("backend/main.py").is_file() {
            return Ok(path);
        }
    }
    let current = std::env::current_dir().map_err(|e| e.to_string())?;
    for candidate in current.ancestors() {
        if candidate.join("backend/main.py").is_file() {
            return Ok(candidate.to_path_buf());
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for candidate in manifest.ancestors() {
        if candidate.join("backend/main.py").is_file() {
            return Ok(candidate.to_path_buf());
        }
    }
    Err("MECH backend source not found. Set MECH_REPO_ROOT to the repository root.".into())
}

fn storage_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    if let Some(app_data) = std::env::var_os("APPDATA") {
        return PathBuf::from(app_data).join("mech-platform/storage/app.db");
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join("Library/Application Support/mech-platform/storage/app.db");
    }
    #[cfg(target_os = "linux")]
    if let Some(config) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(config).join("mech-platform/storage/app.db");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config/mech-platform/storage/app.db");
    }
    PathBuf::from("storage/app.db")
}

fn main() {
    let root = find_repo_root().unwrap_or_else(|error| {
        eprintln!("[MECH] {error}");
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    });
    let storage = AppStorage::open(&storage_path())
        .unwrap_or_else(|error| panic!("MECH storage initialization failed: {error}"));
    let backend_status_state = Arc::new(Mutex::new("STARTING".to_owned()));
    let backend = Arc::new(Mutex::new(None));
    let status_for_thread = Arc::clone(&backend_status_state);
    let backend_for_thread = Arc::clone(&backend);
    let root_for_thread = root.clone();
    std::thread::spawn(move || match BackendSupervisor::start(&root_for_thread) {
        Ok(supervisor) => {
            if let Ok(mut slot) = backend_for_thread.lock() {
                *slot = Some(supervisor);
            }
            if let Ok(mut status) = status_for_thread.lock() {
                *status = "RUNNING".into();
            }
            eprintln!("[MECH] Python scientific compatibility worker is healthy.");
        }
        Err(error) => {
            if let Ok(mut status) = status_for_thread.lock() {
                *status = format!("UNAVAILABLE: {error}");
            }
            eprintln!("[MECH] Python scientific compatibility worker unavailable: {error}");
        }
    });

    let app_state = AppState {
        root,
        storage: Mutex::new(storage),
        client: Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("reqwest client"),
        backend_status: backend_status_state,
        _backend: backend,
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            api_request,
            backend_call,
            backend_status,
            settings_get,
            settings_set,
            settings_reset,
            storage_list,
            storage_upsert,
            storage_remove,
            storage_clear,
            storage_logs
        ])
        .run(tauri::generate_context!())
        .expect("error while running MECH desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_db() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mech-rust-storage-{nonce}.db"))
    }

    #[test]
    fn sqlite_storage_round_trips_existing_workspace_records() {
        let path = temporary_db();
        let storage = AppStorage::open(&path).unwrap();
        let project = json!({"id":"project-1","name":"MECH","path":"C:/work/MECH","createdAt":"2026-01-01T00:00:00.000Z"});
        storage.upsert("projects", &project).unwrap();
        let rows = storage.list("projects").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], "project-1");
        assert_eq!(rows[0]["name"], "MECH");
        assert!(storage.remove("projects", "project-1").unwrap());
        assert!(storage.list("projects").unwrap().is_empty());
        drop(storage);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn settings_patch_preserves_unmodified_nested_defaults() {
        let path = temporary_db();
        let storage = AppStorage::open(&path).unwrap();
        let mut settings = default_settings();
        merge_json(
            &mut settings,
            &json!({"theme":"dark","gpu":{"enabled":false}}),
        );
        storage.set_setting(&settings).unwrap();
        let restored = storage.get_setting().unwrap().unwrap();
        assert_eq!(restored["theme"], "dark");
        assert_eq!(restored["gpu"]["enabled"], false);
        assert_eq!(restored["gpu"]["acceleration"], "auto");
        drop(storage);
        let _ = std::fs::remove_file(&path);
    }
}
