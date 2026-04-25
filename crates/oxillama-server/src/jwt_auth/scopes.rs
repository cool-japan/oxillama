//! JWT scope definitions for OxiLLaMa API authorization.

/// Recognized authorization scopes carried inside JWT tokens.
///
/// Scopes follow the `resource:action` convention used by OAuth 2.0 and the
/// OpenAI API specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Read access to `/v1/chat/completions`.
    ChatRead,
    /// Write (generation) access to `/v1/chat/completions`.
    ChatWrite,
    /// Read access to `/v1/embeddings`.
    EmbedRead,
    /// Read access to `/admin/*` routes.
    AdminRead,
    /// Write (control) access to `/admin/*` routes.
    AdminWrite,
}

impl Scope {
    /// Returns the canonical string representation (`"chat:read"` etc.).
    pub fn as_str(&self) -> &'static str {
        match self {
            Scope::ChatRead => "chat:read",
            Scope::ChatWrite => "chat:write",
            Scope::EmbedRead => "embed:read",
            Scope::AdminRead => "admin:read",
            Scope::AdminWrite => "admin:write",
        }
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Scope {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat:read" => Ok(Scope::ChatRead),
            "chat:write" => Ok(Scope::ChatWrite),
            "embed:read" => Ok(Scope::EmbedRead),
            "admin:read" => Ok(Scope::AdminRead),
            "admin:write" => Ok(Scope::AdminWrite),
            _ => Err(format!("unknown scope: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn scope_round_trip() {
        for (input, expected) in [
            ("chat:read", Scope::ChatRead),
            ("chat:write", Scope::ChatWrite),
            ("embed:read", Scope::EmbedRead),
            ("admin:read", Scope::AdminRead),
            ("admin:write", Scope::AdminWrite),
        ] {
            let parsed = Scope::from_str(input).expect("valid scope string");
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), input);
        }
    }

    #[test]
    fn scope_unknown_returns_err() {
        assert!(Scope::from_str("unknown:scope").is_err());
        assert!(Scope::from_str("").is_err());
    }

    #[test]
    fn scope_display() {
        assert_eq!(Scope::ChatRead.to_string(), "chat:read");
        assert_eq!(Scope::AdminWrite.to_string(), "admin:write");
    }
}
