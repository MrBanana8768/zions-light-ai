FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies including nginx
RUN apt-get update && apt-get install -y wget curl git cmake build-essential libcurl4-openssl-dev nginx && \
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

# Create directories and download model
RUN mkdir -p /app/models /app/data
WORKDIR /app/models
RUN wget https://huggingface.co/Triangle104/Huihui-Qwen3-8B-abliterated-v2-Q4_K_S-GGUF/resolve/main/huihui-qwen3-8b-abliterated-v2-q4_k_s.gguf

# Install Open WebUI
WORKDIR /app
RUN pip install --upgrade pip setuptools wheel && \
    pip install --ignore-installed blinker open-webui

# Create nginx configuration for API authentication
RUN printf 'server {\n\
    listen 8000;\n\
    server_name _;\n\
    \n\
    location / {\n\
        # Check for Authorization header\n\
        set $auth_header $http_authorization;\n\
        if ($auth_header != "Bearer ${API_SECRET_KEY}") {\n\
            return 401 "{\\"error\\": {\\"message\\": \\"Invalid API key\\", \\"type\\": \\"invalid_request_error\\"}}";\n\
        }\n\
        \n\
        # Proxy to llama.cpp server\n\
        proxy_pass http://127.0.0.1:8001;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        \n\
        # Handle CORS for browser requests\n\
        add_header Access-Control-Allow-Origin *;\n\
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";\n\
        add_header Access-Control-Allow-Headers "Authorization, Content-Type";\n\
        \n\
        if ($request_method = OPTIONS) {\n\
            return 204;\n\
        }\n\
    }\n\
}' > /etc/nginx/sites-available/api-auth

# Enable the nginx config
RUN ln -s /etc/nginx/sites-available/api-auth /etc/nginx/sites-enabled/ && \
    rm /etc/nginx/sites-enabled/default

# Create startup script
RUN printf '#!/bin/bash\n\
\n\
if [ -z "$WEBUI_SECRET_KEY" ] || [ -z "$WEBUI_JWT_SECRET_KEY" ]; then\n\
  echo "ERROR: WEBUI_SECRET_KEY and WEBUI_JWT_SECRET_KEY must be set!"\n\
  exit 1\n\
fi\n\
\n\
if [ -z "$API_SECRET_KEY" ]; then\n\
  echo "ERROR: API_SECRET_KEY must be set for API authentication!"\n\
  exit 1\n\
fi\n\
\n\
echo "Starting nginx reverse proxy..."\n\
# Replace placeholder in nginx config with actual API key\n\
sed -i "s/\\${API_SECRET_KEY}/$API_SECRET_KEY/g" /etc/nginx/sites-available/api-auth\n\
nginx -g "daemon off;" &\n\
\n\
echo "Starting Qwen3-8B with llama.cpp on port 8001..."\n\
cd /app/llama.cpp/build\n\
./bin/llama-server -m /app/models/huihui-qwen3-8b-abliterated-v2-q4_k_s.gguf \\\n\
  --host 127.0.0.1 \\\n\
  --port 8001 \\\n\
  --n-gpu-layers 35 \\\n\
  --ctx-size 12288 \\\n\
  --batch-size 512 \\\n\
  --threads 8 \\\n\
  --ctx-shift \\\n\
  --log-disable &\n\
\n\
echo "Waiting for llama.cpp to initialize..."\n\
sleep 60\n\
\n\
echo "Starting Open WebUI on port 3000..."\n\
export DATA_DIR=/app/data\n\
export ENABLE_SIGNUP=${ENABLE_SIGNUP:-true}\n\
export OPENAI_API_BASE_URL=http://127.0.0.1:8001/v1\n\
export OPENAI_API_KEY=dummy\n\
\n\
open-webui serve --host 0.0.0.0 --port 3000\n' > start.sh

RUN chmod +x start.sh

EXPOSE 3000 8000
VOLUME ["/app/data"]
CMD ["/app/start.sh"]