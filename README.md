# vision-inference

Synchronous face and scene inference primitives backed by ONNX Runtime.

## Scope

- Face detection, alignment, and 512-dimensional AuraFace/ArcFace embeddings.
- Face quality gates (size, brightness, blur) folded into the result.
- Passive spoof scoring and optional eye-aspect-ratio (EAR) landmarks.
- 384-dimensional DINOv2 ViT-S/14 scene embeddings.

Not included: databases, vector stores, tenant handling, HTTP APIs, liveness
session state machines, and thresholds for business decisions. Persistence,
API routing, and accept/reject policy belong to the caller.

## Requirements

The crate does not bundle model assets. Supply each ONNX file path through
`FaceConfig` or `SceneConfig`. The target environment must provide the ONNX
Runtime dynamic library; the crate uses `ort` with the `load-dynamic` feature.
Set `ORT_DYLIB_PATH` to the library, or keep `libonnxruntime` on the loader's
search path.

The first `FaceEngine::load` or `SceneEngine::load_standalone` creates one
process-wide ONNX environment. Later loads reuse it. If you already own an
environment, use `FaceEngine::load_with` or `SceneEngine::load(config, &env)`.
`vision_inference::ort` is re-exported for custom setup.

```toml
[dependencies]
# Not published on crates.io: point the git URL at your mirror, or use a local checkout.
vision-inference = { git = "https://example.com/vision-inference.git", tag = "v0.2.0" }
# vision-inference = { path = "/path/to/vision-inference" }
image = "0.25"
```

Replace `tag` with `rev = "<sha>"` if you need an immutable pin.

```bash
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
```

Expected face models are a YuNet detector, an AuraFace (ArcFace-compatible
ResNet100) embedder, a MiniFASNet spoof model, and a PIPNet 68-point landmark
model. The scene model is a DINOv2-small ViT-S/14 ONNX export.

## Usage

Inference is synchronous and the engines serialize sessions behind mutexes.
Async runtimes must call these methods from `spawn_blocking` (or an equivalent
blocking pool) so ONNX work never occupies an async worker.

Paths in the example are relative to the process working directory; pass
absolute paths if the working directory is not fixed.

```rust,no_run
use image::open;
use vision_inference::face::{FaceConfig, FaceEngine};
use vision_inference::scene::{SceneConfig, SceneEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let faces = FaceEngine::load(FaceConfig {
        detector_model: "models/face_detection_yunet_2026may.onnx".into(),
        embedder_model: "models/glintr100.onnx".into(),
        spoof_model: "models/minifasnet_v2_128.onnx".into(),
        landmark_model: Some("models/pipnet_r18_68.onnx".into()),
        model_version: "auraface-r100-v1".into(),
        detector_input_size: 640,
        detector_score_threshold: 0.5,
        detector_iou_threshold: 0.4,
        min_face_size: 64,
        blur_threshold: 40.0,
        brightness_min: 40.0,
        brightness_max: 220.0,
    })?;

    let image = open("face.jpg")?;
    let face = faces.analyze(&image)?;
    assert_eq!(face.embedding.len(), 512);
    // Use faces.analyze_liveness(&image)? for left/right EAR and face geometry.

    let scenes = SceneEngine::load_standalone(SceneConfig {
        model: "models/dinov2_vits14.onnx".into(),
        model_version: "dinov2-small".into(),
        input_size: 224,
        threshold: 0.8,
    })?;

    let scene = scenes.analyze(&image)?;
    assert_eq!(scene.embedding.len(), 384);

    let score = FaceEngine::similarity(&face.embedding, &face.embedding);
    assert!((score - 1.0).abs() < 1e-5);
    Ok(())
}
```

Embeddings are L2-normalized: face vectors are 512-d and scene vectors are
384-d, so cosine similarity is a dot product. `FaceEngine::similarity` and
`SceneEngine::similarity` return that dot product.

`FaceConfig` thresholds only gate face detection and quality; `SceneConfig`'s
`threshold` is exposed via `SceneEngine::threshold()` for the caller to apply.
Neither crate makes a business decision.

## License

Apache-2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for third-party attribution.
