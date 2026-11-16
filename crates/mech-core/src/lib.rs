use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Execution states are deliberately distinct: an unavailable implementation
/// must never be represented as a successful run with empty or fabricated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExecutionStatus {
    NotExecuted,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub model: String,
    pub model_revision: String,
    pub prompt_hash: String,
    pub config_hash: String,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificResult<T> {
    pub status: ExecutionStatus,
    pub provenance: Option<Provenance>,
    pub data: Option<T>,
    pub reason: Option<String>,
}

impl<T> ScientificResult<T> {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self.status {
            ExecutionStatus::NotExecuted | ExecutionStatus::Failed => {
                if self.data.is_some() {
                    return Err("non-successful result must not contain data");
                }
                if self.reason.as_deref().unwrap_or("").trim().is_empty() {
                    return Err("non-successful result requires an explicit reason");
                }
            }
            ExecutionStatus::Running => {
                if self.data.is_some() {
                    return Err("running result must not contain final data");
                }
            }
            ExecutionStatus::Completed => {
                if self.data.is_none() {
                    return Err("completed result requires data");
                }
                if self.provenance.is_none() {
                    return Err("completed result requires provenance");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiProxyRequest {
    #[serde(default = "default_method")]
    pub method: String,
    pub path: String,
    pub body: Option<Value>,
    pub query: Option<BTreeMap<String, Value>>,
    pub headers: Option<BTreeMap<String, String>>,
}

fn default_method() -> String {
    "GET".to_owned()
}

#[derive(Debug, Clone)]
pub struct ValidatedApiRequest {
    pub method: String,
    pub path: String,
    pub query: Vec<(String, String)>,
    pub body: Option<Value>,
    pub headers: Vec<(String, String)>,
}

impl ApiProxyRequest {
    /// Strictly constrain renderer-controlled requests to the local MECH API.
    /// Authorization and transport-level headers are always owned by Rust.
    pub fn validate(self) -> Result<ValidatedApiRequest, String> {
        let method = self.method.to_ascii_uppercase();
        if !matches!(method.as_str(), "GET" | "POST" | "PUT" | "PATCH" | "DELETE") {
            return Err(format!("api request method not allowed: {method}"));
        }
        if !self.path.starts_with('/')
            || self.path.len() > 2048
            || self.path.contains("..")
            || self.path.contains("://")
        {
            return Err("api request path is invalid".to_owned());
        }
        let segments: Vec<&str> = self.path.trim_matches('/').split('/').collect();
        if segments.is_empty() || !matches!(segments[0], "api" | "health") {
            return Err("api request path is outside the allowed API surface".to_owned());
        }
        if self.path.contains('?') || self.path.contains('#') || self.path.contains('\\') {
            return Err(
                "api request path must not contain query, fragment, or backslash".to_owned(),
            );
        }
        if !self
            .path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '~' | '-'))
        {
            return Err("api request path contains unsupported characters".to_owned());
        }
        if let Some(body) = &self.body {
            if !body.is_object() && !body.is_array() && !body.is_null() {
                return Err("api request body must be a JSON object or array".to_owned());
            }
            if body.to_string().len() > 32 * 1024 * 1024 {
                return Err("api request body exceeds 32 MiB".to_owned());
            }
        }
        let mut query = Vec::new();
        if let Some(values) = self.query {
            for (key, value) in values {
                let scalar = match value {
                    Value::String(s) => s,
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => continue,
                    _ => return Err(format!("query parameter {key} must be scalar")),
                };
                query.push((key, scalar));
            }
        }
        let allowed = ["accept", "content-type", "x-request-id", "x-mech-client"];
        let mut headers = Vec::new();
        if let Some(values) = self.headers {
            for (name, value) in values {
                let lower = name.to_ascii_lowercase();
                if lower == "authorization" || lower == "host" || lower == "content-length" {
                    continue;
                }
                if !allowed.contains(&lower.as_str()) {
                    return Err(format!("api request header not allowed: {name}"));
                }
                headers.push((lower, value));
            }
        }
        Ok(ValidatedApiRequest {
            method,
            path: self.path,
            query,
            body: self.body,
            headers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(path: &str) -> ApiProxyRequest {
        ApiProxyRequest {
            method: "POST".into(),
            path: path.into(),
            body: Some(json!({"prompt":"test"})),
            query: None,
            headers: None,
        }
    }

    #[test]
    fn rejects_path_traversal_and_external_urls() {
        assert!(request("/api/../health").validate().is_err());
        assert!(request("https://example.com/api").validate().is_err());
        assert!(request("/other/private").validate().is_err());
    }

    #[test]
    fn strips_renderer_authorization_and_rejects_unknown_headers() {
        let mut req = request("/api/ping");
        req.headers = Some(BTreeMap::from([
            ("Authorization".into(), "attacker-token".into()),
            ("Content-Type".into(), "application/json".into()),
        ]));
        let valid = req.validate().unwrap();
        assert_eq!(
            valid.headers,
            vec![("content-type".into(), "application/json".into())]
        );
        let mut invalid = request("/api/ping");
        invalid.headers = Some(BTreeMap::from([("x-forwarded-host".into(), "evil".into())]));
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn scientific_results_fail_closed() {
        let missing = ScientificResult::<Value> {
            status: ExecutionStatus::NotExecuted,
            provenance: None,
            data: Some(json!({"score": 1.0})),
            reason: Some("engine unavailable".into()),
        };
        assert!(missing.validate().is_err());
        let valid = ScientificResult::<Value> {
            status: ExecutionStatus::NotExecuted,
            provenance: None,
            data: None,
            reason: Some("engine not implemented".into()),
        };
        assert!(valid.validate().is_ok());
    }
}
