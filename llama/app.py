from fastapi import FastAPI, Depends, Body, Query

from llama.application.authenticator import verify_token
from llama.application import examples
from llama.application.model_services import predict, get_first_n_tokens, get_num_tokens
from llama.generation import Dialog, ChatPrediction

app = FastAPI(debug=True)


@app.post("/get_first_n_tokens", dependencies=[Depends(verify_token)])
async def get_num_tokens_of_text(
    text: str = Body(example="This is an example."),
) -> int:
    return get_num_tokens(text=text)


@app.post("/get_first_n_tokens", dependencies=[Depends(verify_token)])
async def get_first_n_tokens_of_text(
    n: int = Query(example=100),
    text: str = Body(example="This is an example."),
) -> str:
    return get_first_n_tokens(text=text, n=n)


@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_nli(
    input_dialogs: list[Dialog] = Body(example=examples.example_dialog),
    temperature: float = Query(0, example=0.6),
    top_p: float = Query(1, example=0.9),
    max_gen_len: int | None = Query(None, example=100),
    logprobs: bool = Query(False, example=False),
) -> list[ChatPrediction]:
    return predict(
        dialogs=input_dialogs,
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
