from fastapi import FastAPI, Depends, Body, Query

from llama.application.authenticator import verify_token
from llama.application import examples
from llama.application.model_services import predict
from llama.generation import Dialog, ChatPrediction

app = FastAPI(debug=True)


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
