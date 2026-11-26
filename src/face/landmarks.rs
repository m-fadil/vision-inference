//! PIPNet 68-point facial landmarks, used for server-side EAR.
//!
//! The ONNX model is the MIT-licensed PIPNet R18 300W/CelebA export. It takes a
//! 256x256 RGB crop with ImageNet normalization and returns five tensors. This
//! implementation uses the landmark's own heatmap cell and offset; PIPNet's
//! optional neighbor averaging is omitted because it only refines the same
//! points and does not change the open/closed eye ratio materially.
//!
//! shortcut: own-cell decode, not neighbor-averaged; upgrade if EAR calibration
//! shows noisy landmark positions.

use image::RgbImage;
use image::imageops::{self, FilterType};
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;

const INPUT_SIZE: u32 = 256;
const FEATURE_SIZE: usize = 8;
const STRIDE: f32 = 32.0;
const LANDMARKS: usize = 68;

pub struct LandmarkDetector {
    session: Session,
    input_name: String,
}

impl LandmarkDetector {
    pub fn load(path: &Path, environment: &Environment) -> Result<Self, String> {
        let session = Session::builder(environment)
            .map_err(|error| format!("failed to create landmark session: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("failed to set landmark optimization level: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| format!("failed to set landmark threads: {error}"))?
            .commit_from_file(path)
            .map_err(|error| {
                format!("failed to load landmark model {}: {error}", path.display())
            })?;
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            session,
            input_name,
        })
    }

    pub fn ear(&mut self, image: &RgbImage, bbox: &[f32; 4]) -> Result<(f32, f32), String> {
        let (crop, crop_width, crop_height, x1, y1) = crop_face(image, bbox);
        let resized = imageops::resize(&crop, INPUT_SIZE, INPUT_SIZE, FilterType::Lanczos3);

        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for (x, y, pixel) in resized.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = (f32::from(pixel[0]) / 255.0 - 0.485) / 0.229;
            input[[0, 1, y as usize, x as usize]] = (f32::from(pixel[1]) / 255.0 - 0.456) / 0.224;
            input[[0, 2, y as usize, x as usize]] = (f32::from(pixel[2]) / 255.0 - 0.406) / 0.225;
        }

        let value = Value::from_array(input).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| format!("landmark inference failed: {error}"))?;

        let cls = output_vec(&outputs, "cls_map")?;
        let offset_x = output_vec(&outputs, "offset_x")?;
        let offset_y = output_vec(&outputs, "offset_y")?;
        let expected = LANDMARKS * FEATURE_SIZE * FEATURE_SIZE;
        if cls.len() != expected || offset_x.len() != expected || offset_y.len() != expected {
            return Err(format!(
                "unexpected landmark output sizes: cls={}, x={}, y={}",
                cls.len(),
                offset_x.len(),
                offset_y.len()
            ));
        }

        let mut points = [(0.0f32, 0.0f32); LANDMARKS];
        for (i, point) in points.iter_mut().enumerate() {
            let base = i * FEATURE_SIZE * FEATURE_SIZE;
            let mut best = 0;
            for j in 1..FEATURE_SIZE * FEATURE_SIZE {
                if cls[base + j] > cls[base + best] {
                    best = j;
                }
            }
            let col = (best % FEATURE_SIZE) as f32;
            let row = (best / FEATURE_SIZE) as f32;
            let x = (col + offset_x[base + best]) / (INPUT_SIZE as f32 / STRIDE);
            let y = (row + offset_y[base + best]) / (INPUT_SIZE as f32 / STRIDE);
            // Map normalized crop coordinates back to the original image.
            *point = (
                x * crop_width as f32 + x1 as f32,
                y * crop_height as f32 + y1 as f32,
            );
        }

        let left = eye_ratio(&points[36..42]);
        let right = eye_ratio(&points[42..48]);
        if !left.is_finite() || !right.is_finite() {
            return Err("landmark EAR is not finite".to_string());
        }
        Ok((left, right))
    }
}

fn output_vec(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Vec<f32>, String> {
    outputs
        .get(name)
        .ok_or_else(|| format!("landmark model has no `{name}` output"))?
        .try_extract_array::<f32>()
        .map(|array| array.iter().copied().collect())
        .map_err(|error| error.to_string())
}

fn crop_face(image: &RgbImage, bbox: &[f32; 4]) -> (RgbImage, u32, u32, u32, u32) {
    let width = (bbox[2] - bbox[0] + 1.0).max(1.0);
    let height = (bbox[3] - bbox[1] + 1.0).max(1.0);
    let pad_x = width * 0.1;
    let pad_y = height * 0.1;

    let x1 = (bbox[0] - pad_x).floor().max(0.0) as u32;
    let y1 = (bbox[1] - pad_y).floor().max(0.0) as u32;
    let x2 = (bbox[2] + pad_x).ceil().min(image.width() as f32) as u32;
    let y2 = (bbox[3] + pad_y).ceil().min(image.height() as f32) as u32;
    let crop_width = (x2.saturating_sub(x1)).max(1);
    let crop_height = (y2.saturating_sub(y1)).max(1);
    let crop = imageops::crop_imm(
        image,
        x1.min(image.width().saturating_sub(1)),
        y1.min(image.height().saturating_sub(1)),
        crop_width.min(image.width()),
        crop_height.min(image.height()),
    )
    .to_image();
    (crop, crop_width, crop_height, x1, y1)
}

fn eye_ratio(points: &[(f32, f32)]) -> f32 {
    let vertical = distance(points[1], points[5]) + distance(points[2], points[4]);
    let horizontal = distance(points[0], points[3]);
    if horizontal <= 1e-6 {
        return 0.0;
    }
    vertical / (2.0 * horizontal)
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_ratio_is_open_for_wide_eye() {
        let points = [
            (0.0, 0.0),
            (1.0, -1.0),
            (2.0, -1.0),
            (3.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
        ];
        let ratio = eye_ratio(&points);
        assert!(ratio > 0.5);
    }

    #[test]
    fn eye_ratio_is_zero_for_closed_eye() {
        let points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (2.0, 0.0),
            (1.0, 0.0),
        ];
        assert!(eye_ratio(&points) < 1e-6);
    }

    #[test]
    fn crop_face_pads_equally_on_both_axes() {
        let image = RgbImage::new(200, 200);
        let (_, width, height, x1, y1) = crop_face(&image, &[10.0, 10.0, 110.0, 110.0]);
        assert_eq!((x1, y1), (0, 0));
        assert_eq!(width, height);
    }
}
