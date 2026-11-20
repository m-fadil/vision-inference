//! ArcFace embedding for an aligned 112x112 RGB face.
//!
//! The model expects `(pixel - 127.5) / 127.5` in RGB channel order and returns
//! an unnormalized 512-dimensional vector. We L2-normalize it so cosine
//! similarity is a dot product.

use image::RgbImage;
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;

const INPUT_SIZE: usize = 112;
const EMBEDDING_DIM: usize = 512;

pub struct Embedder {
    session: Session,
    input_name: String,
}

impl Embedder {
    pub fn load(path: &Path, environment: &Environment) -> Result<Self, String> {
        let session = Session::builder(environment)
            .map_err(|error| format!("failed to create embedder session: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("failed to set embedder optimization level: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| format!("failed to set embedder threads: {error}"))?
            .commit_from_file(path)
            .map_err(|error| format!("failed to load embedder {}: {error}", path.display()))?;
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            session,
            input_name,
        })
    }

    pub fn embedding(&mut self, image: &RgbImage) -> Result<Vec<f32>, String> {
        if image.dimensions() != (INPUT_SIZE as u32, INPUT_SIZE as u32) {
            return Err(format!(
                "ArcFace requires a {INPUT_SIZE}x{INPUT_SIZE} input, got {}x{}",
                image.width(),
                image.height()
            ));
        }
        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        let data = input
            .as_slice_memory_order_mut()
            .ok_or_else(|| "failed to access embedder input tensor".to_string())?;
        let plane = INPUT_SIZE * INPUT_SIZE;
        for (i, [r, g, b]) in image.as_raw().as_chunks::<3>().0.iter().enumerate() {
            data[i] = (f32::from(*r) - 127.5) / 127.5;
            data[plane + i] = (f32::from(*g) - 127.5) / 127.5;
            data[2 * plane + i] = (f32::from(*b) - 127.5) / 127.5;
        }

        let value = Value::from_array(input).map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => value])
            .map_err(|error| format!("embedder inference failed: {error}"))?;
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|error| error.to_string())?;
        let mut embedding: Vec<f32> = output.iter().copied().collect();
        if embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "unexpected ArcFace output length: {} (expected {EMBEDDING_DIM})",
                embedding.len()
            ));
        }
        if !crate::math::l2_normalize(&mut embedding) {
            return Err("ArcFace produced a degenerate embedding".to_string());
        }
        Ok(embedding)
    }
}
