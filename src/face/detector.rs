//! YuNet face detector (three stride levels, dynamic input).
//!
//! The model consumes a BGR image, padded to a multiple of 32, and exposes
//! `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`, and `kps_{8,16,32}`.
//! Scores are `sqrt(cls * obj)`. Boxes and landmarks are decoded relative to
//! each grid cell. YuNet names the eyes/mouth corners from the subject's side;
//! in image coordinates they run left-to-right, which is the ArcFace template
//! order used by `align.rs`.

use image::{RgbImage, imageops};
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;

const STRIDES: [usize; 3] = [8, 16, 32];
const PAD_DIVISOR: u32 = 32;

#[derive(Debug, Clone)]
pub struct Detection {
    /// Absolute pixel coordinates in the source image.
    pub bbox: [f32; 4],
    /// Image-left eye, image-right eye, nose, image-left mouth corner,
    /// image-right mouth corner.
    pub landmarks: [(f32, f32); 5],
    pub score: f32,
}

pub struct Detector {
    session: Session,
    input_name: String,
    /// Longest side of the detector input. Smaller images are not upscaled.
    max_side: u32,
    score_threshold: f32,
    iou_threshold: f32,
}

impl Detector {
    pub fn load(
        path: &Path,
        environment: &Environment,
        max_side: u32,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Self, String> {
        let session = Session::builder(environment)
            .map_err(|error| format!("failed to create detector session: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("failed to set detector optimization level: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| format!("failed to set detector threads: {error}"))?
            .commit_from_file(path)
            .map_err(|error| format!("failed to load detector {}: {error}", path.display()))?;
        let input_name = session.inputs()[0].name().to_string();

        for stride in STRIDES {
            for prefix in ["cls", "obj", "bbox", "kps"] {
                let name = format!("{prefix}_{stride}");
                if !session.outputs().iter().any(|output| output.name() == name) {
                    return Err(format!("model is missing {name}"));
                }
            }
        }

        Ok(Self {
            session,
            input_name,
            max_side: max_side.max(PAD_DIVISOR),
            score_threshold,
            iou_threshold,
        })
    }

    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>, String> {
        let (input_image, scale) = prepare(image, self.max_side)?;
        let input = Self::input_tensor(&input_image)?;
        let value = Value::from_array(input).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| format!("detector inference failed: {error}"))?;

        let (padded_width, padded_height) = input_image.dimensions();
        let mut candidates = Vec::new();
        for stride in STRIDES {
            let columns = padded_width as usize / stride;
            let rows = padded_height as usize / stride;
            let cells = rows * columns;
            let scores = output_values(&outputs, &format!("cls_{stride}"), cells)?;
            let objects = output_values(&outputs, &format!("obj_{stride}"), cells)?;
            let bboxes = output_values(&outputs, &format!("bbox_{stride}"), cells * 4)?;
            let keypoints = output_values(&outputs, &format!("kps_{stride}"), cells * 10)?;

            for row in 0..rows {
                for column in 0..columns {
                    let index = row * columns + column;
                    let class_score = scores[index].clamp(0.0, 1.0);
                    let object_score = objects[index].clamp(0.0, 1.0);
                    let score = (class_score * object_score).sqrt();
                    if score < self.score_threshold {
                        continue;
                    }
                    if let Some(detection) = decode_cell(
                        stride,
                        row,
                        column,
                        &bboxes[index * 4..index * 4 + 4],
                        &keypoints[index * 10..index * 10 + 10],
                        score,
                        scale,
                    ) {
                        candidates.push(detection);
                    }
                }
            }
        }

        Ok(non_maximum_suppression(candidates, self.iou_threshold))
    }

    fn input_tensor(image: &RgbImage) -> Result<Array4<f32>, String> {
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let mut input = Array4::<f32>::zeros((1, 3, height, width));
        let data = input
            .as_slice_memory_order_mut()
            .ok_or_else(|| "failed to access detector input tensor".to_string())?;
        let plane = width * height;
        for (index, [r, g, b]) in image.as_raw().as_chunks::<3>().0.iter().enumerate() {
            // OpenCV builds the YuNet blob without swapRB: raw BGR in 0..255.
            data[index] = f32::from(*b);
            data[plane + index] = f32::from(*g);
            data[2 * plane + index] = f32::from(*r);
        }
        Ok(input)
    }
}

/// Resize to at most `max_side` and pad right/bottom with zeros to a multiple
/// of 32. Returns the prepared image and the resize scale (1.0 = no resize).
fn prepare(image: &RgbImage, max_side: u32) -> Result<(RgbImage, f32), String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("cannot detect on an empty image".to_string());
    }
    let ratio = (max_side as f32 / width.max(height) as f32).min(1.0);
    let resized_width = ((width as f32 * ratio).round() as u32).max(1);
    let resized_height = ((height as f32 * ratio).round() as u32).max(1);
    let resized = if (resized_width, resized_height) == (width, height) {
        image.clone()
    } else {
        imageops::resize(
            image,
            resized_width,
            resized_height,
            imageops::FilterType::CatmullRom,
        )
    };
    let padded_width = resized_width.div_ceil(PAD_DIVISOR) * PAD_DIVISOR;
    let padded_height = resized_height.div_ceil(PAD_DIVISOR) * PAD_DIVISOR;
    let mut padded = RgbImage::new(padded_width, padded_height);
    imageops::overlay(&mut padded, &resized, 0, 0);
    Ok((padded, resized_width as f32 / width as f32))
}

fn decode_cell(
    stride: usize,
    row: usize,
    column: usize,
    bbox: &[f32],
    keypoints: &[f32],
    score: f32,
    scale: f32,
) -> Option<Detection> {
    let stride = stride as f32;
    let center_x = (column as f32 + bbox[0]) * stride;
    let center_y = (row as f32 + bbox[1]) * stride;
    let width = bbox[2].exp() * stride;
    let height = bbox[3].exp() * stride;
    if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
        return None;
    }

    let point = |index: usize| {
        (
            (keypoints[index * 2] + column as f32) * stride / scale,
            (keypoints[index * 2 + 1] + row as f32) * stride / scale,
        )
    };
    let landmarks = [
        point(0), // eye on the image-left
        point(1), // eye on the image-right
        point(2), // nose tip
        point(3), // mouth corner on the image-left
        point(4), // mouth corner on the image-right
    ];

    Some(Detection {
        bbox: [
            (center_x - width / 2.0) / scale,
            (center_y - height / 2.0) / scale,
            (center_x + width / 2.0) / scale,
            (center_y + height / 2.0) / scale,
        ],
        landmarks,
        score,
    })
}

fn output_values(
    outputs: &SessionOutputs<'_>,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let value = outputs
        .get(name)
        .ok_or_else(|| format!("model output {name} is missing"))?;
    let array = value
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?;
    let values: Vec<f32> = array.iter().copied().collect();
    if values.len() != expected {
        return Err(format!(
            "{name} has {} values, expected {expected}",
            values.len()
        ));
    }
    Ok(values)
}

fn non_maximum_suppression(mut faces: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    faces.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
    let mut kept: Vec<Detection> = Vec::with_capacity(faces.len());
    for face in faces {
        if kept
            .iter()
            .all(|other| iou(&face.bbox, &other.bbox) <= threshold)
        {
            kept.push(face);
        }
    }
    kept
}

fn iou(left: &[f32; 4], right: &[f32; 4]) -> f32 {
    let x1 = left[0].max(right[0]);
    let y1 = left[1].max(right[1]);
    let x2 = left[2].min(right[2]);
    let y2 = left[3].min(right[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if intersection <= 0.0 {
        return 0.0;
    }
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    intersection / (left_area + right_area - intersection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn prepare_resizes_and_pads_to_divisor() {
        let image = RgbImage::new(100, 70);
        let (prepared, scale) = prepare(&image, 64).unwrap();
        assert_eq!(prepared.dimensions(), (64, 64));
        assert!((scale - 0.64).abs() < 1e-6);
    }

    #[test]
    fn prepare_does_not_upscale_small_images() {
        let image = RgbImage::new(30, 20);
        let (prepared, scale) = prepare(&image, 640).unwrap();
        assert_eq!(prepared.dimensions(), (32, 32));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn decode_cell_keeps_image_left_to_right_landmark_order() {
        let bbox = [0.25, -0.25, 0.0, 0.0];
        let kps = [
            0.01, 0.02, // eye on the image-left
            0.11, 0.12, // eye on the image-right
            0.21, 0.22, // nose
            0.31, 0.32, // mouth corner on the image-left
            0.41, 0.42, // mouth corner on the image-right
        ];
        let detection = decode_cell(8, 2, 3, &bbox, &kps, 0.9, 1.0).unwrap();
        assert_close(detection.bbox[0], 22.0);
        assert_close(detection.bbox[1], 10.0);
        assert_close(detection.bbox[2], 30.0);
        assert_close(detection.bbox[3], 18.0);
        assert_close(detection.landmarks[0].0, 24.08);
        assert_close(detection.landmarks[1].0, 24.88);
        assert_close(detection.landmarks[2].0, 25.68);
        assert_close(detection.landmarks[3].0, 26.48);
        assert_close(detection.landmarks[4].0, 27.28);
    }

    #[test]
    fn decode_cell_rejects_non_finite_box() {
        let bbox = [0.0, 0.0, f32::INFINITY, 0.0];
        let kps = [0.0; 10];
        assert!(decode_cell(8, 0, 0, &bbox, &kps, 0.9, 1.0).is_none());
    }

    #[test]
    fn iou_is_one_for_identical_boxes() {
        let box_ = [10.0, 20.0, 30.0, 40.0];
        assert!((iou(&box_, &box_) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_is_zero_for_disjoint_boxes() {
        assert_eq!(iou(&[0.0, 0.0, 1.0, 1.0], &[2.0, 2.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn nms_keeps_highest_score() {
        let candidates = vec![
            Detection {
                bbox: [0.0, 0.0, 10.0, 10.0],
                landmarks: [(0.0, 0.0); 5],
                score: 0.8,
            },
            Detection {
                bbox: [1.0, 1.0, 9.0, 9.0],
                landmarks: [(0.0, 0.0); 5],
                score: 0.9,
            },
        ];
        let kept = non_maximum_suppression(candidates, 0.4);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].score, 0.9);
    }
}
