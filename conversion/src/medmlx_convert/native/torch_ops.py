"""Translate installed ATen signatures into the core's portable graph contract."""

from medmlx_core.graph.schema import Argument, Call, Op

OPERATORS = {
    "aten.conv2d.default": Op.CONV2,
    "aten.conv3d.default": Op.CONV3,
    "aten.conv_transpose2d.input": Op.DECONV2,
    "aten.conv_transpose3d.input": Op.DECONV3,
    "aten.batch_norm.default": Op.BATCH_NORM,
    "aten._native_batch_norm_legit_no_training.default": Op.EVAL_BATCH_NORM,
    "aten.instance_norm.default": Op.INSTANCE_NORM,
    "aten.relu.default": Op.RELU,
    "aten.leaky_relu.default": Op.LEAKY_RELU,
    "aten.avg_pool2d.default": Op.AVG_POOL2,
    "aten.upsample_bilinear2d.vec": Op.BILINEAR,
    "aten.cat.default": Op.CAT,
    "aten.slice.Tensor": Op.SLICE,
    "aten.log_softmax.int": Op.LOG_SOFTMAX,
    "aten.add.Tensor": Op.ADD,
    "aten.sub.Tensor": Op.SUB,
    "aten.mul.Tensor": Op.MUL,
    "aten.div.Tensor": Op.DIV,
    "aten.view.default": Op.RESHAPE,
    "aten.reshape.default": Op.RESHAPE,
    "aten.split.Tensor": Op.SPLIT,
    "aten._to_copy.default": Op.CAST,
    "aten.to.dtype": Op.CAST,
    "aten._assert_tensor_metadata.default": Op.ASSERT_META,
}
_NAMES = {"self": "input", "a": "input", "dim": "axis",
          "split_size": "chunk_size", "size": "shape"}


def lower_call(name: str, target: str, arguments: tuple[Argument, ...]) -> Call:
    """Bind source defaults, reject unsupported metadata, and emit canonical arguments.

    Shape/dtype casts retain their meaning; source execution/device options are
    admitted only where the fixed CPU inference contract makes them irrelevant.
    Core owns the one required-parameter list for every portable operation.
    """
    op = OPERATORS[target]
    source = {argument.name: argument.value for argument in arguments}
    if op == Op.CAST and any(source.get(key) is not None
                             for key in ("layout", "device", "memory_format")):
        raise ValueError("Device/layout casts are unsupported")
    if op == Op.ASSERT_META and (
        source.get("stride") is not None or source.get("device") not in (None, "cpu")
        or source.get("layout") not in (None, "strided")
    ):
        raise ValueError("Unsupported source tensor metadata assertion")
    normalized = {_NAMES.get(key, key): value for key, value in source.items()}
    canonical = tuple(Argument(key, value) for key, value in normalized.items()
                      if key in op.parameters)
    return Call(name, op, canonical)
