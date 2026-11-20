# Third-party notices

## face_id-rs

The SCRFD and ArcFace inference pipeline in `src/face/detector.rs`,
`src/face/embedder.rs`, and `src/face/align.rs` is an internal reimplementation
informed by the public algorithms and tensor layouts in
[`RuurdBijlsma/face_id-rs`](https://github.com/RuurdBijlsma/face_id-rs)
(Apache License 2.0). The `face_id` crate is no longer linked or vendored.

Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

## DINOv2 ViT-S/14 scene model

`dinov2_vits14.onnx` is used for scene embeddings. The model weights are from
[`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small)
(Apache License 2.0). The ONNX export is from
[`sefaburak/dinov2-small-onnx`](https://huggingface.co/sefaburak/dinov2-small-onnx)
(Apache License 2.0). Model assets are not distributed with this crate; the
caller supplies the ONNX file through `SceneConfig`.

DINOv2: <https://github.com/facebookresearch/dinov2>

Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>
