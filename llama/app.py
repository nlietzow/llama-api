from fastapi import FastAPI, Depends, Query

from llama.application.authenticator import verify_token
from llama.application.models import (
    TextInputModel,
    TextWithNumTokensModel,
    InferenceInputModel,
)
from llama.application.model_services import (
    predict,
    get_first_n_tokens,
    get_num_tokens,
    TextWithNumTokens,
)
from llama.generation import ChatPrediction

app = FastAPI(debug=True)


@app.post(
    "/get_num_tokens",
    dependencies=[Depends(verify_token)],
    response_model=TextWithNumTokensModel,
)
async def get_num_tokens_of_text(text: TextInputModel) -> TextWithNumTokens:
    return get_num_tokens(text=text.text)


@app.post(
    "/get_first_n_tokens",
    dependencies=[Depends(verify_token)],
    response_model=TextWithNumTokensModel,
)
async def get_first_n_tokens_of_text(
    text: TextInputModel, n: int = Query(example=100)
) -> TextWithNumTokens:
    return get_first_n_tokens(text=text.text, n=n)


@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_nli(
    input_dialogs: InferenceInputModel,
    temperature: float = Query(0, example=0.7),
    top_p: float = Query(1, example=1),
    max_gen_len: int | None = Query(None, example=100),
    logprobs: bool = Query(False, example=False),
) -> list[ChatPrediction]:
    return predict(
        dialogs=input_dialogs.dialogs,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
        logprobs=logprobs,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llama.app:app",
        host="0.0.0.0",
        port=8000,
    )
