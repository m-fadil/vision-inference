//! DINOv2 scene embedding.
//!
//! The ONNX graph takes a 224x224 RGB image normalized with ImageNet
//! statistics and returns a 384-dimensional CLS vector. Vectors are
//! L2-normalized so cosine similarity maps directly to similarity.
//!
//! Model: facebook/dinov2-small (Apache-2.0), exported to ONNX. The ONNX file
//! is fetched separately and is not committed to the repository.

use image::imageops::{self, FilterType};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;

const EMBEDDING_DIM: usize = 384;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct SceneConfig {
    pub model: PathBuf,
    pub model_version: String,
    pub input_size: u32,
    pub threshold: f32,
}

#[derive(Debug, Clone)]
pub struct SceneAnalysis {
    pub embedding: Vec<f32>,
    pub model_version: String,
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("image error: {0}")]
    Image(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("inference error: {0}")]
    Inference(String),
}

pub struct SceneEngine {
    config: SceneConfig,
    _environment: Environment,
    session: Mutex<Session>,
    input_name: String,
}

// Fail the build if the shared session stops being thread-safe.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SceneEngine>();
};

impl SceneEngine {
    pub fn load(config: SceneConfig, environment: &Environment) -> Result<Self, SceneError> {
        let session = Session::builder(environment)
            .map_err(|error| SceneError::Model(format!("failed to create scene session: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                SceneError::Model(format!("failed to set scene optimization level: {error}"))
            })?
            .with_intra_threads(2)
            .map_err(|error| SceneError::Model(format!("failed to set scene threads: {error}")))?
            .commit_from_file(&config.model)
            .map_err(|error| {
                SceneError::Model(format!(
                    "failed to load scene model {}: {error}",
                    config.model.display()
                ))
            })?;
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            config,
            _environment: environment.clone(),
            session: Mutex::new(session),
            input_name,
        })
    }

    pub fn analyze(&self, image: &DynamicImage) -> Result<SceneAnalysis, SceneError> {
        let size = self.config.input_size;
        let resized = preprocess(image, size)?;
        let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
        let data = input
            .as_slice_memory_order_mut()
            .ok_or_else(|| SceneError::Inference("failed to access scene input".into()))?;
        let plane = size as usize * size as usize;
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..3 {
                data[channel * plane + index] =
                    (f32::from(pixel[channel]) / 255.0 - MEAN[channel]) / STD[channel];
            }
        }

        let value =
            Value::from_array(input).map_err(|error| SceneError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| SceneError::Inference("scene lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| SceneError::Inference(format!("scene inference failed: {error}")))?;
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|error| SceneError::Inference(error.to_string()))?;
        let mut embedding: Vec<f32> = output.iter().copied().collect();
        if embedding.len() != EMBEDDING_DIM {
            return Err(SceneError::Inference(format!(
                "unexpected DINOv2 output length: {} (expected {EMBEDDING_DIM})",
                embedding.len()
            )));
        }
        normalize(&mut embedding)?;
        Ok(SceneAnalysis {
            embedding,
            model_version: self.config.model_version.clone(),
        })
    }

    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
            .clamp(-1.0, 1.0)
    }

    pub fn threshold(&self) -> f32 {
        self.config.threshold
    }

    pub fn config(&self) -> &SceneConfig {
        &self.config
    }
}

/// Resize the short side to 256 and center-crop 224. This preserves the
/// aspect ratio instead of squashing 16:9 camera frames into a square.
fn preprocess(image: &DynamicImage, size: u32) -> Result<RgbImage, SceneError> {
    if size == 0 {
        return Err(SceneError::Image(
            "scene input size must be positive".into(),
        ));
    }
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(SceneError::Image("empty scene image".into()));
    }
    let short = width.min(height);
    let resized_short = size.saturating_mul(8) / 7;
    let scale = resized_short as f32 / short as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(size);
    let resized_height = ((height as f32 * scale).round() as u32).max(size);
    let resized = imageops::resize(
        &image.to_rgb8(),
        resized_width,
        resized_height,
        FilterType::Lanczos3,
    );
    let left = (resized_width - size) / 2;
    let top = (resized_height - size) / 2;
    Ok(imageops::crop_imm(&resized, left, top, size, size).to_image())
}

fn normalize(vector: &mut [f32]) -> Result<(), SceneError> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-12 {
        return Err(SceneError::Inference(
            "DINOv2 produced a degenerate embedding".into(),
        ));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_produces_unit_vector() {
        let mut vector = [3.0, 4.0];
        normalize(&mut vector).unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_rejects_zero_vector() {
        assert!(normalize(&mut [0.0, 0.0]).is_err());
    }

    #[test]
    fn similarity_is_cosine_for_normalized_vectors() {
        assert!((SceneEngine::similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(SceneEngine::similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
    #[test]
    fn preprocess_preserves_aspect_ratio_and_output_size() {
        let image = DynamicImage::new_rgb8(320, 180);
        let output = preprocess(&image, 224).unwrap();
        assert_eq!(output.dimensions(), (224, 224));
    }
}
