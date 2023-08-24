


1. Create Instanz
2. git clone https://github.com/nlietzow/llama-api.git
2. cd llama-api
3. bash download.sh -> model will be downloaded to llama-api/llama-2-7b-chat
4. docker build . -t app:latest
5. docker run -d -p 8000:8000 -v ${PWD}/llama-2-7b-chat:/code/llama-2-7b-chat --gpus all --restart always app:latest