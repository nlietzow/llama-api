from pydantic import BaseModel

example_dialog = [
    [
        {
            "role": "system",
            "content": "Always answer with emojis",
        },
        {
            "role": "user",
            "content": "How to go from Beijing to NY?",
        },
    ],
]


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
