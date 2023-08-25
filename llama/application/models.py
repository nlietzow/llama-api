from pydantic import BaseModel, field_validator

from llama.application import settings
from llama.generation import Dialog, Message

example_dialog: Dialog = [
    Message(
        role="system",
        content="Always answer with emojis",
    ),
    Message(
        role="user",
        content="How to go from Beijing to NY?",
    ),
]


class InferenceInputModel(BaseModel):
    dialogs: list[Dialog]

    @field_validator("dialogs")
    def validate_dialogs(cls, v):
        if len(v) > settings.max_batch_size:
            raise ValueError(
                f"Number of dialogs ({len(v)}) exceeds max batch size ({settings.max_batch_size})."
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "dialogs": [example_dialog],
            }
        }


class TextInputModel(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "This is an example.",
            }
        }


class TextWithNumTokensModel(TextInputModel):
    num_tokens: int

    class Config:
        schema_extra = {
            "example": {
                "text": "This is an example.",
                "num_tokens": 5,
            }
        }
