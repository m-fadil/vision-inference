//! Face detection, quality gating, alignment and ArcFace embedding.
//!
//! `FaceEngine` is deliberately synchronous: callers on an async runtime should
//! run it on a blocking thread so ONNX inference never blocks a worker.

use image::{DynamicImage, GenericImageView, RgbImage};
use ort::environment::Environment;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;

mod align;
mod detector;
mod embedder;
mod landmarks;
mod spoof;
use crate::math::cosine_similarity;
use detector::Detector;
use embedder::Embedder;
use landmarks::LandmarkDetector;
pub use spoof::SpoofAnalysis;
use spoof::SpoofDetector;

#[derive(Debug, Clone)]
pub struct FaceConfig {
    pub detector_model: PathBuf,
    pub embedder_model: PathBuf,
    pub spoof_model: PathBuf,
    /// Required only by [`FaceEngine::analyze_liveness`].
    pub landmark_model: Option<PathBuf>,
    pub model_version: String,
    pub detector_input_size: u32,
    pub detector_score_threshold: f32,
    pub detector_iou_threshold: f32,
    pub min_face_size: u32,
    pub blur_threshold: f64,
    pub brightness_min: f64,
    pub brightness_max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Quality {
    pub passed: bool,
    pub score: f32,
    pub blur: f32,
    pub brightness: f32,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FaceAnalysis {
    pub embedding: Vec<f32>,
    /// Normalized `[x1, y1, x2, y2]` coordinates in the range 0..1.
    pub bbox: [f32; 4],
    pub quality: Quality,
    pub spoof: SpoofAnalysis,
    pub ear: Option<f32>,
    pub model_version: String,
}

#[derive(Debug, Error)]
pub enum FaceError {
    #[error("no face detected")]
    NoFace,
    #[error("multiple faces detected: {0}")]
    MultipleFaces(usize),
    #[error("low quality face")]
    LowQuality(Quality),
    /// A caller-supplied byte stream could not be decoded. The engine itself
    /// accepts already-decoded [`DynamicImage`] values.
    #[error("image decode failed: {0}")]
    Image(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("inference error: {0}")]
    Inference(String),
}

pub struct FaceEngine {
    config: FaceConfig,
    _environment: Environment,
    detector: Mutex<Detector>,
    embedder: Mutex<Embedder>,
    spoof: Mutex<SpoofDetector>,
    landmark: Option<Mutex<LandmarkDetector>>,
}

// Shared across threads; fail the build if the sessions stop being thread-safe.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FaceEngine>();
};

impl FaceEngine {
    pub fn load(config: FaceConfig) -> Result<Self, FaceError> {
        let environment =
            crate::runtime::environment().map_err(|error| FaceError::Model(error.to_string()))?;
        Self::load_with(config, environment)
    }

    /// Load with an existing ONNX Runtime environment. The caller must use the
    /// single environment allowed per process.
    pub fn load_with(config: FaceConfig, environment: &Environment) -> Result<Self, FaceError> {
        let (w, h) = (config.detector_input_size, config.detector_input_size);
        let detector = Detector::load(
            &config.detector_model,
            environment,
            (w, h),
            config.detector_score_threshold,
            config.detector_iou_threshold,
        )
        .map_err(FaceError::Model)?;

        let embedder =
            Embedder::load(&config.embedder_model, environment).map_err(FaceError::Model)?;

        let spoof =
            SpoofDetector::load(&config.spoof_model, environment).map_err(FaceError::Model)?;
        let landmark = config
            .landmark_model
            .as_ref()
            .map(|path| LandmarkDetector::load(path, environment))
            .transpose()
            .map_err(FaceError::Model)?;

        Ok(Self {
            config,
            _environment: environment.clone(),
            detector: Mutex::new(detector),
            embedder: Mutex::new(embedder),
            spoof: Mutex::new(spoof),
            landmark: landmark.map(Mutex::new),
        })
    }

    pub fn analyze(&self, image: &DynamicImage) -> Result<FaceAnalysis, FaceError> {
        self.analyze_inner(image, false)
    }

    pub fn analyze_liveness(&self, image: &DynamicImage) -> Result<FaceAnalysis, FaceError> {
        self.analyze_inner(image, true)
    }

    fn analyze_inner(
        &self,
        image: &DynamicImage,
        with_ear: bool,
    ) -> Result<FaceAnalysis, FaceError> {
        let (width, height) = image.dimensions();
        let rgb = image.to_rgb8();

        let detections = self
            .detector
            .lock()
            .map_err(|_| FaceError::Inference("detector lock poisoned".into()))?
            .detect(&rgb)
            .map_err(|e| FaceError::Inference(format!("detection failed: {e}")))?;

        match detections.len() {
            0 => return Err(FaceError::NoFace),
            1 => {}
            n => return Err(FaceError::MultipleFaces(n)),
        }

        let face = detections.into_iter().next().expect("one detection");
        let bbox = [
            face.bbox[0] / width as f32,
            face.bbox[1] / height as f32,
            face.bbox[2] / width as f32,
            face.bbox[3] / height as f32,
        ];
        let pixel_bbox = face.bbox;
        // Use the shorter side of the box in pixels as the face size.
        let face_size = f64::from(
            (pixel_bbox[2] - pixel_bbox[0])
                .min(pixel_bbox[3] - pixel_bbox[1])
                .max(0.0),
        );

        let (brightness, blur) = face_stats(&rgb, &pixel_bbox);

        let quality = evaluate_quality(&self.config, face.score, face_size, brightness, blur);
        if !quality.passed {
            return Err(FaceError::LowQuality(quality));
        }

        let spoof = self
            .spoof
            .lock()
            .map_err(|_| FaceError::Inference("spoof lock poisoned".into()))?
            .score(&rgb, &pixel_bbox)
            .map_err(FaceError::Inference)?;

        let ear = if with_ear {
            let landmark = self
                .landmark
                .as_ref()
                .ok_or_else(|| FaceError::Model("landmark model is not configured".into()))?;
            Some(
                landmark
                    .lock()
                    .map_err(|_| FaceError::Inference("landmark lock poisoned".into()))?
                    .ear(&rgb, &pixel_bbox)
                    .map_err(FaceError::Inference)?,
            )
        } else {
            None
        };

        let aligned = align::norm_crop(&rgb, &face.landmarks)
            .ok_or_else(|| FaceError::Inference("invalid face landmarks".into()))?;

        let embedding = self
            .embedder
            .lock()
            .map_err(|_| FaceError::Inference("embedder lock poisoned".into()))?
            .embedding(&aligned)
            .map_err(|e| FaceError::Inference(format!("embedding failed: {e}")))?;

        Ok(FaceAnalysis {
            embedding,
            bbox,
            quality,
            spoof,
            ear,
            model_version: self.config.model_version.clone(),
        })
    }

    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b)
    }

    pub fn environment(&self) -> &Environment {
        &self._environment
    }

    pub fn config(&self) -> &FaceConfig {
        &self.config
    }
}

fn evaluate_quality(
    config: &FaceConfig,
    detection_score: f32,
    face_size: f64,
    brightness: f64,
    blur: f64,
) -> Quality {
    let min_size = f64::from(config.min_face_size);
    let mut reasons = Vec::new();
    if face_size < min_size {
        reasons.push("face_too_small".to_string());
    }
    if brightness < config.brightness_min {
        reasons.push("too_dark".to_string());
    }
    if brightness > config.brightness_max {
        reasons.push("too_bright".to_string());
    }
    if blur < config.blur_threshold {
        reasons.push("blurry".to_string());
    }

    let size_factor = if min_size > 0.0 {
        (face_size / min_size).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let brightness_factor = if brightness < config.brightness_min {
        brightness / config.brightness_min.max(1e-6)
    } else if brightness > config.brightness_max {
        (255.0 - brightness) / (255.0 - config.brightness_max).max(1e-6)
    } else {
        1.0
    }
    .clamp(0.0, 1.0);
    let blur_factor = if config.blur_threshold > 0.0 {
        (blur / config.blur_threshold).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let detection_factor = f64::from(detection_score.clamp(0.0, 1.0));
    let score =
        ((size_factor + brightness_factor + blur_factor + detection_factor) / 4.0).clamp(0.0, 1.0);

    Quality {
        passed: reasons.is_empty(),
        score: score as f32,
        blur: blur as f32,
        brightness: brightness as f32,
        reasons,
    }
}

/// Mean luma and variance of the Laplacian (3x3) over the face region.
fn face_stats(rgb: &RgbImage, bbox: &[f32; 4]) -> (f64, f64) {
    let (iw, ih) = rgb.dimensions();
    let clamp = |v: f32, max: u32| v.max(0.0).min(max as f32) as u32;
    let x1 = clamp(bbox[0].floor(), iw);
    let y1 = clamp(bbox[1].floor(), ih);
    let x2 = clamp(bbox[2].ceil(), iw);
    let y2 = clamp(bbox[3].ceil(), ih);
    if x2 <= x1 || y2 <= y1 {
        return (0.0, 0.0);
    }
    let (w, h) = ((x2 - x1) as usize, (y2 - y1) as usize);
    let luma_row = |y: u32| -> Vec<f64> {
        (x1..x2)
            .map(|x| {
                let p = rgb.get_pixel(x, y);
                0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
            })
            .collect()
    };

    let mut above = luma_row(y1);
    let mut brightness = above.iter().sum::<f64>();
    if h == 1 {
        return (brightness / w as f64, 0.0);
    }
    let mut current = luma_row(y1 + 1);
    brightness += current.iter().sum::<f64>();

    if w < 3 || h < 3 {
        return (brightness / (w * h) as f64, 0.0);
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut n = 0u64;
    for y in (y1 + 2)..y2 {
        let below = luma_row(y);
        brightness += below.iter().sum::<f64>();
        for x in 1..w - 1 {
            let lap = above[x] + below[x] + current[x - 1] + current[x + 1] - 4.0 * current[x];
            sum += lap;
            sum_sq += lap * lap;
            n += 1;
        }
        above = current;
        current = below;
    }
    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64 - mean * mean).max(0.0);
    (brightness / (w * h) as f64, variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FaceConfig {
        FaceConfig {
            detector_model: PathBuf::from("det.onnx"),
            embedder_model: PathBuf::from("rec.onnx"),
            spoof_model: PathBuf::from("spoof.onnx"),
            landmark_model: Some(PathBuf::from("landmark.onnx")),
            model_version: "test".into(),
            detector_input_size: 640,
            detector_score_threshold: 0.5,
            detector_iou_threshold: 0.4,
            min_face_size: 100,
            blur_threshold: 20.0,
            brightness_min: 25.0,
            brightness_max: 235.0,
        }
    }

    #[test]
    fn similarity_identical_is_one() {
        let v = [0.6, 0.8, 0.0];
        assert!((FaceEngine::similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_orthogonal_is_zero() {
        assert_eq!(FaceEngine::similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn similarity_length_mismatch_is_zero() {
        assert_eq!(FaceEngine::similarity(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn quality_accepts_good_face() {
        let q = evaluate_quality(&config(), 0.95, 200.0, 128.0, 60.0);
        assert!(q.passed);
        assert!(q.reasons.is_empty());
        assert!(q.score > 0.8);
    }

    #[test]
    fn quality_collects_failure_reasons() {
        let small_dark = evaluate_quality(&config(), 0.95, 50.0, 10.0, 60.0);
        assert!(!small_dark.passed);
        assert_eq!(small_dark.reasons, ["face_too_small", "too_dark"]);

        let bright_blurry = evaluate_quality(&config(), 0.95, 200.0, 250.0, 5.0);
        assert!(!bright_blurry.passed);
        assert_eq!(bright_blurry.reasons, ["too_bright", "blurry"]);
    }

    #[test]
    fn face_stats_handles_constant_face_without_a_full_plane() {
        let image = RgbImage::from_pixel(20, 20, image::Rgb([100, 100, 100]));
        let (brightness, blur) = face_stats(&image, &[0.0, 0.0, 19.0, 19.0]);
        assert!((brightness - 100.0).abs() < 1e-6);
        assert_eq!(blur, 0.0);
    }
}
