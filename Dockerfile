FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV LOCAL_RANK=0
ENV WORLD_SIZE=1
ENV NPROC_PER_NODE=1

WORKDIR /code

COPY llama llama
COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e .

CMD ["uvicorn", "llama.app:app", "--host", "0.0.0.0", "--port", "8000"]
