# vision-inference

Synchronous face and scene inference primitives backed by ONNX Runtime.

## Scope

- Face detection, alignment, and 512-dimensional ArcFace embeddings.
- Face quality gates (size, brightness, blur) folded into the result.
- Passive spoof scoring and optional eye-aspect-ratio (EAR) landmarks.
- 384-dimensional DINOv2 ViT-S/14 scene embeddings.

Not included: databases, vector stores, tenant handling, HTTP APIs, liveness
session state machines, and thresholds for business decisions. Persistence,
API routing, and accept/reject policy belong to the caller.

## Requirements

The crate does not bundle model assets. Supply each ONNX file path through
`FaceConfig` or `SceneConfig`. The target environment must also provide the
ONNX Runtime dynamic library; the crate uses `ort` with the `load-dynamic`
feature, so set `ORT_DYLIB_PATH` to the library and depend on a compatible
`ort` version (`2.0.0-rc.15`).

```toml
[dependencies]
vision-inference = { path = "../vision-inference" }
image = "0.25"
```

```bash
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
```

Expected face models are an SCRFD detector, an ArcFace embedder, a MiniFASNet
spoof model, and a PIPNet 68-point landmark model. The scene model is a
DINOv2-small ViT-S/14 ONNX export.

## Usage

Inference is synchronous and the engines serialize sessions behind mutexes.
Async runtimes must call these methods from `spawn_blocking` (or an equivalent
blocking pool) so ONNX work never occupies an async worker.

```rust
use image::open;
use vision_inference::face::{FaceConfig, FaceEngine};
use vision_inference::scene::{SceneConfig, SceneEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let faces = FaceEngine::load(FaceConfig {
        detector_model: "models/2.5g_bnkps.onnx".into(),
        embedder_model: "models/w600k_r50.onnx".into(),
        spoof_model: "models/minifasnet_v2_128.onnx".into(),
        landmark_model: "models/pipnet_r18_68.onnx".into(),
        model_version: "buffalo_l".into(),
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
    // Use faces.analyze_liveness(&image)? for an additional EAR value.

    let scenes = SceneEngine::load(
        SceneConfig {
            model: "models/dinov2_vits14.onnx".into(),
            model_version: "dinov2-small".into(),
            input_size: 224,
            threshold: 0.8,
        },
        &faces.environment(),
    )?;

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

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party
attribution.
