//! Process-wide ONNX Runtime environment.
//!
//! `ort` 2.0 allows only one environment per process and its singleton slot is
//! never cleared. Cache the environment here so repeated engine loads and
//! retries reuse it instead of failing with "only one environment is allowed".

use std::env;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ort::environment::Environment;
use thiserror::Error;

static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Error)]
pub(crate) enum RuntimeError {
    #[error("failed to load ONNX Runtime dylib {}: {message}", path.display())]
    Load { path: PathBuf, message: String },
    #[error("failed to initialize ONNX Runtime: {0}")]
    Init(String),
    #[error("ONNX Runtime initialization lock poisoned")]
    Lock,
}

pub(crate) fn environment() -> Result<&'static Environment, RuntimeError> {
    if let Some(environment) = ENVIRONMENT.get() {
        return Ok(environment);
    }

    let _guard = INIT_LOCK.lock().map_err(|_| RuntimeError::Lock)?;
    if let Some(environment) = ENVIRONMENT.get() {
        return Ok(environment);
    }

    let path = env::var_os("ORT_DYLIB_PATH")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "{}onnxruntime{}",
                env::consts::DLL_PREFIX,
                env::consts::DLL_SUFFIX
            ))
        });
    let builder = ort::init_from(&path).map_err(|error| RuntimeError::Load {
        path: path.clone(),
        message: error.to_string(),
    })?;
    let environment = builder
        .with_name("vision-inference")
        .build()
        .map_err(|error| RuntimeError::Init(error.to_string()))?;
    ENVIRONMENT
        .set(environment)
        .map_err(|_| RuntimeError::Lock)?;
    ENVIRONMENT.get().ok_or(RuntimeError::Lock)
}
