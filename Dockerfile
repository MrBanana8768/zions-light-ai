FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies including nginx
RUN apt-get update && apt-get install -y wget curl git cmake build-essential libcurl4-openssl-dev nginx jq && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# Create CUDA stub symlinks and fix linking
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

# Build llama.cpp with CMake and CUDA support
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp
RUN mkdir build && cd build && \
    cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" && \
    cmake --build . --config Release --parallel 8

# Create directories and download models
RUN mkdir -p /app/models /app/data
WORKDIR /app/models
RUN wget https://huggingface.co/Triangle104/Huihui-Qwen3-8B-abliterated-v2-Q4_K_S-GGUF/resolve/main/huihui-qwen3-8b-abliterated-v2-q4_k_s.gguf -O qwen3
RUN wget https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF/resolve/main/solar-10.7b-instruct-v1.0-uncensored.Q5_K_S.gguf -O solar.gguf

# Install Open WebUI
WORKDIR /app
RUN pip install --upgrade pip setuptools wheel && \
    pip install --ignore-installed blinker open-webui

# Create nginx configuration for multi-model routing
RUN printf 'upstream model1_backend {\n\
    server 127.0.0.1:9001;\n\
}\n\
\n\
upstream model2_backend {\n\
    server 127.0.0.1:9002;\n\
}\n\
\n\
server {\n\
    listen 9000;\n\
    server_name _;\n\
    \n\
    location /health {\n\
        return 200 "{\\"status\\": \\"healthy\\", \\"models\\": [\\"qwen3\\", \\"solar\\"]}";\n\
        add_header Content-Type application/json always;\n\
    }\n\
    \n\
    location /v1/models {\n\
        default_type application/json;\n\
        return 200 "{\n\
            \\"object\\": \\"list\\",\n\
            \\"data\\": [\n\
                {\n\
                    \\"id\\": \\"qwen3\\",\n\
                    \\"object\\": \\"model\\",\n\
                    \\"created\\": 1234567890,\n\
                    \\"owned_by\\": \\"local\\"\n\
                },\n\
                {\n\
                    \\"id\\": \\"solar\\",\n\
                    \\"object\\": \\"model\\",\n\
                    \\"created\\": 1234567890,\n\
                    \\"owned_by\\": \\"local\\"\n\
                }\n\
            ]\n\
        }";\n\
    }\n\
    \n\
    location /v1/chat/completions {\n\
        set $backend model1_backend;\n\
        \n\
        if ($http_x_model = "solar") {\n\
            set $backend model2_backend;\n\
        }\n\
        \n\
        proxy_pass http://$backend;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        \n\
        add_header Access-Control-Allow-Origin * always;\n\
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;\n\
        add_header Access-Control-Allow-Headers "Authorization, Content-Type, X-Model" always;\n\
        \n\
        if ($request_method = OPTIONS) {\n\
            return 204;\n\
        }\n\
    }\n\
}' > /etc/nginx/sites-available/multi-model

# Add include directive to main nginx.conf
RUN sed -i '/http {/a\    include /etc/nginx/sites-enabled/*;' /etc/nginx/nginx.conf

# Enable the nginx config
RUN ln -s /etc/nginx/sites-available/multi-model /etc/nginx/sites-enabled/ && \
    rm /etc/nginx/sites-enabled/default

# Create startup script with dual model support
RUN printf '#!/bin/bash\n\
\n\
if [ -z "$WEBUI_SECRET_KEY" ] || [ -z "$WEBUI_JWT_SECRET_KEY" ] || [ -z "$API_SECRET_KEY" ]; then\n\
  echo "ERROR: WEBUI_SECRET_KEY, WEBUI_JWT_SECRET_KEY, and API_SECRET_KEY must be set!"\n\
  exit 1\n\
fi\n\
\n\
echo "Starting nginx reverse proxy..."\n\
sed -i "s/\\${API_SECRET_KEY}/$API_SECRET_KEY/g" /etc/nginx/sites-available/multi-model\n\
nginx -g "daemon off;" &\n\
\n\
echo "Starting Model 1 (Qwen3-8B) on port 9001..."\n\
cd /app/llama.cpp/build\n\
./bin/llama-server -m /app/models/qwen3.gguf \\\n\
  --host 127.0.0.1 \\\n\
  --port 9001 \\\n\
  --n-gpu-layers ${MODEL1_GPU_LAYERS:-12} \\\n\
  --ctx-size ${MODEL1_CTX_SIZE:-12000} \\\n\
  --batch-size ${MODEL1_BATCH_SIZE:-256} \\\n\
  --threads ${MODEL1_THREADS:-4} \\\n\
  --context-shift &\n\
\n\
echo "Starting Model 2 on port 9002..."\n\
./bin/llama-server -m /app/models/solar.gguf \\\n\
  --host 127.0.0.1 \\\n\
  --port 9002 \\\n\
  --n-gpu-layers ${MODEL2_GPU_LAYERS:-20} \\\n\
  --ctx-size ${MODEL2_CTX_SIZE:-20000} \\\n\
  --batch-size ${MODEL2_BATCH_SIZE:-256} \\\n\
  --threads ${MODEL2_THREADS:-4} \\\n\
  --context-shift &\n\
\n\
echo "Waiting for models to initialize..."\n\
sleep 90\n\
\n\
echo "Starting Open WebUI with model discovery..."\n\
export DATA_DIR=/app/data\n\
export ENABLE_SIGNUP=${ENABLE_SIGNUP:-true}\n\
export OPENAI_API_BASE_URL=http://127.0.0.1:9000/v1\n\
export OPENAI_API_KEY=$API_SECRET_KEY\n\
\n\
open-webui serve --host 0.0.0.0 --port 3000\n' > start.sh

RUN chmod +x start.sh

EXPOSE 3000 9000
VOLUME ["/app/data"]
CMD ["/app/start.sh"]