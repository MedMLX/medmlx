# Lazy stride-2 transpose convolution

`conv_transpose3d_ncdhw` dispatches float32 kernel-2/stride-2 convolution with
zero padding and output padding to the existing `deconv2x_ncdhw` operator. It
computes eight matrix-product phases with the original Torch weight layout and
interleaves them into the output. Other geometries and dtypes use the native
MLX operation. No evaluation boundary is added to model graphs.

On MLX 0.32.3 and an M1 Max, the native path can return corrupted values when a
lazy decoder adds bias and a skip tensor to a large output. Isolated evaluation
of the convolution hides the failure. Run the checkpoint-free NumPy oracle:

```sh
python scripts/reproduce_transpose_consumer.py
```

Default command-buffer limits reproduced relative L2 errors of 0.77-0.80 in
the final lazy call; the isolated call matched exactly. Setting
`MLX_MAX_MB_PER_BUFFER=100000 MLX_MAX_OPS_PER_BUFFER=100000` made all three
calls exact. These overrides are diagnostic controls only.

The MLX v0.32.3 source revision is `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`.
Its [Metal convolution implementation](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/conv.cpp#L1001-L1012)
adds `out_phase_view` to `copies`, which are subsequently registered as
temporaries. Those views share the public output buffer.
[CommandEncoder::end_encoding](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp#L445-L471)
removes temporary buffers from the output set before registering cross-encoder
fences. This is the inferred cause of the demonstrated graph corruption from
the pinned source; an upstream C++ repair has not been built or qualified here.

The regression checks the public core operation followed by a lazy skip
addition against Torch, at the unchanged `dot` numerical budget, with repeated
48-cubed inputs. Released SegResNet checkpoint parity remains a separate gate:
removing this corruption does not establish full-model or clinical qualification.
