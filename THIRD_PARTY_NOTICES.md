# Third-party notices

## YuNet face detector

`face_detection_yunet_2026may.onnx` is the YuNet detector from
[`opencv/opencv_zoo`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
The model files in that directory are MIT-licensed. The detector
post-processing in `src/face/detector.rs` follows OpenCV's public YuNet decode.

MIT License: <https://opensource.org/license/mit>

## AuraFace face embedder

`glintr100.onnx` is the AuraFace v1 ArcFace ResNet100 embedder from
[`fal/AuraFace-v1`](https://huggingface.co/fal/AuraFace-v1) (Apache License 2.0).
The model card states that it was trained on commercially available data for
commercial use. The repository also contains an InsightFace SCRFD file; this
crate does not use that file.

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
