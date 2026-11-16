use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;

const BACKEND_ADDR: &str = "127.0.0.1:8000";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("MECH backend source was not found under {0}")]
    BackendMissing(PathBuf),
    #[error("could not start Python backend: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("Python backend exited before becoming healthy (status: {0})")]
    Exited(String),
    #[error("Python backend did not become healthy within {0} seconds")]
    StartupTimeout(u64),
}

/// Owns the legacy scientific worker while the Rust research runtime is built.
/// An already-running healthy service is reused and is never killed on drop.
pub struct BackendSupervisor {
    child: Option<Child>,
}

impl BackendSupervisor {
    pub fn start(repo_root: &Path) -> Result<Self, RuntimeError> {
        let entry = repo_root.join("backend").join("main.py");
        if !entry.is_file() {
            return Err(RuntimeError::BackendMissing(entry));
        }
        if probe_health() {
            return Ok(Self { child: None });
        }

        let python = resolve_python(repo_root);
        let python_path =
            std::env::join_paths([repo_root.to_path_buf(), repo_root.join("backend")]).map_err(
                |e| RuntimeError::Spawn(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)),
            )?;
        let mut child = Command::new(python)
            .arg(&entry)
            .current_dir(repo_root)
            .env("MECH_BIND_HOST", "127.0.0.1")
            .env("PYTHONPATH", python_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let started = Instant::now();
        loop {
            if probe_health() {
                return Ok(Self { child: Some(child) });
            }
            if let Some(status) = child.try_wait()? {
                return Err(RuntimeError::Exited(status.to_string()));
            }
            if started.elapsed() >= STARTUP_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err(RuntimeError::StartupTimeout(STARTUP_TIMEOUT.as_secs()));
            }
            thread::sleep(Duration::from_millis(250));
        }
    }
}

impl Drop for BackendSupervisor {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn resolve_python(repo_root: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("MECH_PYTHON") {
        return PathBuf::from(path);
    }
    for candidate in [
        repo_root.join(".venv/Scripts/python.exe"),
        repo_root.join("venv/Scripts/python.exe"),
        repo_root.join(".venv/bin/python"),
        repo_root.join("venv/bin/python"),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn probe_health() -> bool {
    let address: SocketAddr = BACKEND_ADDR.parse().expect("static socket address");
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(300)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = Vec::new();
    if stream.take(4096).read_to_end(&mut response).is_err() {
        return false;
    }
    is_mech_health_response(&response)
}

fn is_mech_health_response(response: &[u8]) -> bool {
    let response = String::from_utf8_lossy(response);
    let Some((headers, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    let status_line = headers.lines().next().unwrap_or_default();
    let status_ok =
        status_line.starts_with("HTTP/1.1 200") || status_line.starts_with("HTTP/1.0 200");
    status_ok && body.contains("\"status\":\"healthy\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_accepts_mech_health_payloads() {
        assert!(is_mech_health_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n{\"status\":\"healthy\"}"
        ));
        assert!(!is_mech_health_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}"
        ));
    }

    #[test]
    fn reports_missing_backend_instead_of_starting_an_unrelated_process() {
        let missing = PathBuf::from("target/nonexistent-mech-root");
        let result = BackendSupervisor::start(&missing);
        assert!(matches!(result, Err(RuntimeError::BackendMissing(_))));
    }
}
