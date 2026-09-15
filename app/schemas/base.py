"""Base class for Pydantic models bound to an LLM's structured-output tool
schema.

Some models (gpt-oss included) emit `null` for a list field that has
nothing in it, instead of `[]` or omitting the key. Groq validates tool
calls strictly against the JSON schema it was given, and a bare `list[...]`
annotation produces a schema that rejects null — so the whole tool call
gets bounced with a 400 before it ever reaches our code.

The fix has two parts: widen list-typed fields to `list[...] | None` so the
generated schema accepts null (done per-model, since it changes each
field's annotation), and normalize any None back to `[]` here so the rest
of the codebase can keep assuming a list field is never actually None.
"""

import types
import typing

from pydantic import BaseModel, model_validator


class LLMSchema(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _none_lists_to_empty(cls, data):
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            if name in data and data[name] is None and _is_list_annotation(field.annotation):
                data[name] = []
        return data


def _is_list_annotation(annotation) -> bool:
    origin = typing.get_origin(annotation)
    if origin is list:
        return True
    if origin is typing.Union or origin is types.UnionType:
        return any(_is_list_annotation(arg) for arg in typing.get_args(annotation))
    return False
