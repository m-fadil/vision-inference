//! SCRFD face detector (640x640, three stride levels).
//!
//! This is the minimum inference path needed by `FaceEngine`: letterbox the
//! image, run the ONNX graph, decode score/bbox/keypoint tensors, and apply
//! class-agnostic NMS. The configured model must expose
//! `score_{8,16,32}`, `bbox_{8,16,32}`, and `kps_{8,16,32}` outputs.

use image::{RgbImage, imageops};
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;

const STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Debug, Clone)]
pub struct Detection {
    /// Absolute pixel coordinates in the source image.
    pub bbox: [f32; 4],
    pub landmarks: [(f32, f32); 5],
    pub score: f32,
}

pub struct Detector {
    session: Session,
    input_name: String,
    input_size: (u32, u32),
    score_threshold: f32,
    iou_threshold: f32,
    levels: Vec<Level>,
}

struct Level {
    stride: usize,
    score_name: String,
    bbox_name: String,
    kps_name: String,
    anchors: Vec<(f32, f32)>,
}

struct Preprocess {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}

impl Detector {
    pub fn load(
        path: &Path,
        environment: &Environment,
        input_size: (u32, u32),
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

        let mut levels = Vec::with_capacity(STRIDES.len());
        for stride in STRIDES {
            let score_name = format!("score_{stride}");
            let bbox_name = format!("bbox_{stride}");
            let kps_name = format!("kps_{stride}");
            let score = session
                .outputs()
                .iter()
                .find(|output| output.name() == score_name)
                .ok_or_else(|| format!("model is missing {score_name}"))?;
            if !session
                .outputs()
                .iter()
                .any(|output| output.name() == bbox_name)
            {
                return Err(format!("model is missing {bbox_name}"));
            }
            if !session
                .outputs()
                .iter()
                .any(|output| output.name() == kps_name)
            {
                return Err(format!("model is missing {kps_name}"));
            }
            let shape = score
                .dtype()
                .tensor_shape()
                .ok_or_else(|| format!("{score_name} has no static shape"))?;
            let total = if shape.len() >= 2 {
                shape[shape.len() - 2]
            } else {
                return Err(format!("{score_name} has an unexpected shape {shape:?}"));
            };
            if total <= 0 {
                return Err(format!("{score_name} has no anchors: {shape:?}"));
            }
            let anchors = generate_anchors(input_size, stride, total as usize)?;
            levels.push(Level {
                stride,
                score_name,
                bbox_name,
                kps_name,
                anchors,
            });
        }

        Ok(Self {
            session,
            input_name,
            input_size,
            score_threshold,
            iou_threshold,
            levels,
        })
    }

    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>, String> {
        let (input_image, preprocess) = self.preprocess(image)?;
        let input = Self::input_tensor(&input_image)?;
        let value = Value::from_array(input).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| format!("detector inference failed: {error}"))?;

        let mut candidates = Vec::new();
        for level in &self.levels {
            let anchors = level.anchors.len();
            let scores = output_values(&outputs, &level.score_name, anchors)?;
            let bboxes = output_values(&outputs, &level.bbox_name, anchors * 4)?;
            let keypoints = output_values(&outputs, &level.kps_name, anchors * 10)?;
            let stride = level.stride as f32;

            for (index, &(anchor_x, anchor_y)) in level.anchors.iter().enumerate() {
                let score = scores[index];
                if score < self.score_threshold {
                    continue;
                }
                let bbox = &bboxes[index * 4..index * 4 + 4];
                let x1 = decode(
                    -bbox[0],
                    anchor_x,
                    stride,
                    preprocess.offset_x,
                    preprocess.scale_x,
                );
                let y1 = decode(
                    -bbox[1],
                    anchor_y,
                    stride,
                    preprocess.offset_y,
                    preprocess.scale_y,
                );
                let x2 = decode(
                    bbox[2],
                    anchor_x,
                    stride,
                    preprocess.offset_x,
                    preprocess.scale_x,
                );
                let y2 = decode(
                    bbox[3],
                    anchor_y,
                    stride,
                    preprocess.offset_y,
                    preprocess.scale_y,
                );

                let mut landmarks = [(0.0f32, 0.0f32); 5];
                for (point, landmark) in landmarks.iter_mut().enumerate() {
                    let offset = index * 10 + point * 2;
                    landmark.0 = (anchor_x + keypoints[offset] * stride - preprocess.offset_x)
                        / preprocess.scale_x;
                    landmark.1 = (anchor_y + keypoints[offset + 1] * stride - preprocess.offset_y)
                        / preprocess.scale_y;
                }
                candidates.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    landmarks,
                    score,
                });
            }
        }

        Ok(non_maximum_suppression(candidates, self.iou_threshold))
    }

    fn preprocess(&self, image: &RgbImage) -> Result<(RgbImage, Preprocess), String> {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err("cannot detect on an empty image".to_string());
        }
        let (input_width, input_height) = self.input_size;
        let ratio = (input_width as f32 / width as f32).min(input_height as f32 / height as f32);
        let resized_width = ((width as f32 * ratio).round() as u32).max(1);
        let resized_height = ((height as f32 * ratio).round() as u32).max(1);
        let resized = imageops::resize(
            image,
            resized_width,
            resized_height,
            imageops::FilterType::CatmullRom,
        );
        let offset_x = (input_width - resized_width) as f32 / 2.0;
        let offset_y = (input_height - resized_height) as f32 / 2.0;
        let mut padded = RgbImage::new(input_width, input_height);
        imageops::overlay(&mut padded, &resized, offset_x as i64, offset_y as i64);
        Ok((
            padded,
            Preprocess {
                scale_x: resized_width as f32 / width as f32,
                scale_y: resized_height as f32 / height as f32,
                offset_x,
                offset_y,
            },
        ))
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
            data[index] = (f32::from(*r) - 127.5) / 128.0;
            data[plane + index] = (f32::from(*g) - 127.5) / 128.0;
            data[2 * plane + index] = (f32::from(*b) - 127.5) / 128.0;
        }
        Ok(input)
    }
}

fn generate_anchors(
    input_size: (u32, u32),
    stride: usize,
    total: usize,
) -> Result<Vec<(f32, f32)>, String> {
    let grid = (input_size.0 as usize / stride) * (input_size.1 as usize / stride);
    if grid == 0 || !total.is_multiple_of(grid) {
        return Err(format!(
            "cannot derive anchors from {total} values and input {input_size:?} stride {stride}"
        ));
    }
    let anchors_per_cell = total / grid;
    let width = input_size.0 as usize / stride;
    let mut anchors = Vec::with_capacity(total);
    for pixel in 0..grid {
        let x = (pixel % width) as f32 * stride as f32;
        let y = (pixel / width) as f32 * stride as f32;
        anchors.extend(std::iter::repeat_n((x, y), anchors_per_cell));
    }
    Ok(anchors)
}

fn output_values(
    outputs: &SessionOutputs<'_>,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let array = outputs[name]
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

fn decode(value: f32, anchor: f32, stride: f32, offset: f32, scale: f32) -> f32 {
    (anchor + value * stride - offset) / scale
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

    #[test]
    fn anchors_cover_grid_and_repeat_per_cell() {
        let anchors = generate_anchors((640, 640), 16, 3200).unwrap();
        assert_eq!(anchors.len(), 3200);
        assert_eq!(anchors[0], (0.0, 0.0));
        assert_eq!(anchors[1], (0.0, 0.0));
        assert_eq!(anchors[2], (16.0, 0.0));
        assert_eq!(anchors[3199], (624.0, 624.0));
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
