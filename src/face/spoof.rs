//! Passive face anti-spoofing with MiniFASNetV2 (128x128, binary).
//!
//! Model: facenox/face-antispoof-onnx, an Apache-2.0 ONNX export of
//! MiniFASNetV2. The reference preprocessing crops a square around the face
//! with a 1.5x margin, pads out-of-bounds pixels with REFLECT_101, resizes to
//! 128x128 RGB, and normalizes to [0, 1].
//!
//! The exported graph returns raw logits: `[real, spoof]`.

use image::RgbImage;
use image::imageops::{self, FilterType};
use ndarray::{Array4, Ix2};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use serde::Serialize;
use std::path::Path;

const INPUT_SIZE: u32 = 128;
const CROP_MARGIN: f32 = 1.5;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpoofAnalysis {
    pub live: f32,
    pub spoof: f32,
}

impl SpoofAnalysis {
    /// Probability that the face is a spoof (print or replay).
    pub fn score(&self) -> f32 {
        self.spoof
    }

    pub fn is_spoof(&self, threshold: f32) -> bool {
        self.score() >= threshold
    }
}

pub struct SpoofDetector {
    session: Session,
    input_name: String,
}

impl SpoofDetector {
    pub fn load(path: &Path, environment: &Environment) -> Result<Self, String> {
        let session = Session::builder(environment)
            .map_err(|error| format!("failed to create spoof session: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("failed to set spoof optimization level: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| format!("failed to set spoof threads: {error}"))?
            .commit_from_file(path)
            .map_err(|error| format!("failed to load spoof model {}: {error}", path.display()))?;
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            session,
            input_name,
        })
    }

    pub fn score(&mut self, image: &RgbImage, bbox: &[f32; 4]) -> Result<SpoofAnalysis, String> {
        let crop = crop_square(image, bbox);
        let resized = imageops::resize(&crop, INPUT_SIZE, INPUT_SIZE, FilterType::Lanczos3);

        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for (x, y, pixel) in resized.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = f32::from(pixel[0]) / 255.0;
            input[[0, 1, y as usize, x as usize]] = f32::from(pixel[1]) / 255.0;
            input[[0, 2, y as usize, x as usize]] = f32::from(pixel[2]) / 255.0;
        }

        let value = Value::from_array(input).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| format!("spoof inference failed: {error}"))?;
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|error| error.to_string())?
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|error| error.to_string())?;

        if output.shape() != [1, 2] {
            return Err(format!(
                "unexpected spoof output shape: {:?}",
                output.shape()
            ));
        }

        let probabilities = softmax([output[[0, 0]], output[[0, 1]]]);
        Ok(SpoofAnalysis {
            live: probabilities[0],
            spoof: probabilities[1],
        })
    }
}

fn crop_square(image: &RgbImage, bbox: &[f32; 4]) -> RgbImage {
    let box_width = (bbox[2] - bbox[0]).max(1.0);
    let box_height = (bbox[3] - bbox[1]).max(1.0);
    let side = (box_width.max(box_height) * CROP_MARGIN).round() as i32;
    let side = side.max(1);

    let center_x = bbox[0] + box_width * 0.5;
    let center_y = bbox[1] + box_height * 0.5;
    let left = (center_x - side as f32 * 0.5).round() as i32;
    let top = (center_y - side as f32 * 0.5).round() as i32;

    let mut crop = RgbImage::new(side as u32, side as u32);
    for y in 0..side {
        for x in 0..side {
            let source_x = reflect(left + x, image.width());
            let source_y = reflect(top + y, image.height());
            crop.put_pixel(x as u32, y as u32, *image.get_pixel(source_x, source_y));
        }
    }
    crop
}

/// OpenCV `BORDER_REFLECT_101`: `abc` -> `... cba | abc | cba ...`.
fn reflect(index: i32, len: u32) -> u32 {
    if len <= 1 {
        return 0;
    }
    let len = len as i32;
    let mut index = index;
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * len - 2 - index;
        }
    }
    index as u32
}

fn softmax(logits: [f32; 2]) -> [f32; 2] {
    let max = logits[0].max(logits[1]);
    let exp = logits.map(|value| (value - max).exp());
    let sum = exp[0] + exp[1];
    exp.map(|value| value / sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let probabilities = softmax([1.0, 3.0]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[1] > probabilities[0]);
    }

    #[test]
    fn softmax_handles_large_logits() {
        let probabilities = softmax([1000.0, 1002.0]);
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spoof_score_is_spoof_probability() {
        let spoof = SpoofAnalysis {
            live: 0.1,
            spoof: 0.9,
        };
        assert!((spoof.score() - 0.9).abs() < 1e-6);
        assert!(spoof.is_spoof(0.5));
        assert!(!spoof.is_spoof(0.95));
    }

    #[test]
    fn reflect_matches_opencv_reflect_101() {
        assert_eq!(reflect(-1, 6), 1);
        assert_eq!(reflect(-2, 6), 2);
        assert_eq!(reflect(6, 6), 4);
        assert_eq!(reflect(7, 6), 3);
        assert_eq!(reflect(0, 0), 0);
    }
}
