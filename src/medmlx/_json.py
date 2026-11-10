"""JSON key contracts shared by single-case and cohort workers."""

from typing import cast


def require_string_keys(value: object) -> None:
    """Reject keys JSON would coerce, including inside arrays and nested objects."""

    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError("Model run JSON object keys must be strings")
            require_string_keys(item)
    elif isinstance(value, list | tuple):
        for item in cast(list[object] | tuple[object, ...], value):
            require_string_keys(item)

