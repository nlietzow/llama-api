FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /code



COPY llama llama
COPY tokenizer.model tokenizer.model
COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e .


CMD ["torchrun", "--nproc_per_node", "4", "llama/app.py"]
