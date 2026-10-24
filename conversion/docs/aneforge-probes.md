# ANEForge probes

Measured on 2026-10-12, Apple M1 Max, macOS 27.0.1. These isolated probes use
ANEForge revision `896c4943c5ebf811c0e815f8d1381fd1e157fa7d`, run from a separate
checkout so that no MedMLX recipe or model asset is modified.

| Original released network | Input | ANEForge result |
|---|---|---|
| MONAI FlexibleUNet, endoscopic tools | 1×3×736×480 | Executes; max logit error 0.489 versus gate 0.001. Argmax unchanged on the synthetic fixture. Failed. |
| MONAI SegResNet, renal structures | 1×3×96×96×96 | ONNX export succeeds; ANEForge import rejects the 5D convolution input because this frontend expects 4D NCHW. Blocked before inference. |

References are upstream PyTorch CPU FP32 with the exact checkpoint revisions and
seeded inputs used by the maintained recipes. ONNX uses opset 13, as ANEForge's
importer recommends. ANEForge uses FP16 and `approx_resize=False, opt=0`.
The FP32 parity gates were not relaxed. Neither probe qualifies a production
replacement or an application speedup. A single unchanged synthetic mask is
not evidence of segmentation accuracy on medical images. The SegResNet result
describes the current ONNX frontend, not a hardware impossibility claim.

The Core AI/Core ML recipe suite remains separate and passed all 24 tests.
