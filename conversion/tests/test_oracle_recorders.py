"""Host-only validation of optional TensorFlow oracle tensor boundaries."""

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.record_keras_references import FloatArray, model_array, require_model


@dataclass
class EagerResult:
    value: object

    def numpy(self) -> object:
        return self.value


@dataclass
class Inference:
    result: object

    def __call__(self, inputs: FloatArray, *, training: bool) -> object:
        return self.result


@given(st.lists(st.floats(width=32, allow_nan=False, allow_infinity=False), max_size=32))
def test_oracle_array_validation_preserves_fp32_values_and_identity(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float32)
    assert model_array(Inference(EagerResult(array)), array) is array


def test_oracle_boundary_rejects_wrong_input_output_and_non_models() -> None:
    image = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="input"):
        model_array(Inference(EagerResult(image)), image.astype(np.float64))
    with pytest.raises(TypeError, match="eager tensor"):
        model_array(Inference(image), image)
    with pytest.raises(TypeError, match="output"):
        model_array(Inference(EagerResult(image.astype(np.float64))), image)
    with pytest.raises(TypeError, match="Keras model"):
        require_model(Inference(EagerResult(image)))
