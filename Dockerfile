FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV RANK=0

WORKDIR /code

COPY llama llama
COPY tokenizer.model tokenizer.model
COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e .

CMD ["uvicorn", "llama.app:app", "--host", "0.0.0.0", "--port", "8000"]
