#!/bin/bash
# Runs INSIDE the image for `wordban.py e2e`: real OpenWebUI -> real compactor -> the recording vLLM stand-in.
#   $1 = compactor dir (/priv/compactor-src/compactor, or /opt/compactor for the image's own)   $2 = model id
# Mounts: /wb (this tool, ro), /tok (tokenizer, ro), /priv (trimmed webui.db, value, probes; ro)
CDIR=$1; MODEL=$2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p /tmp/e2e/openwebui /tmp/e2e/logs
cp /priv/e2e.db /tmp/e2e/openwebui/webui.db
: > /tmp/e2e/stub.jsonl
/opt/vllm-venv/bin/python /wb/e2e/stub.py /tmp/e2e/stub.jsonl "$MODEL" /priv/probes.json > /tmp/e2e/logs/stub.log 2>&1 &
echo "E2E-COMPACTOR $CDIR main.py md5 $(md5sum $CDIR/main.py | cut -c1-12)"
( cd $CDIR && env VLLM_URL=http://127.0.0.1:8000 MODEL_REPO="$MODEL" \
    DATA_DIR=/tmp/e2e/openwebui COMPACTOR_SELFTEST_ON_BOOT=false COMPACTOR_FACTS_EXTRACTION=false \
    COMPACTOR_BACKUP_ENABLED=false COMPACTOR_ADMIN_BIND=127.0.0.1 \
    /opt/compactor-venv/bin/uvicorn main:app --host 127.0.0.1 --port 8080 > /tmp/e2e/logs/compactor.log 2>&1 & )
( cd /app && env DATA_DIR=/tmp/e2e/openwebui OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 ENABLE_OPENAI_API=true \
    ENABLE_OLLAMA_API=false WEBUI_SECRET_KEY=wb-e2e-secret DATABASE_ENABLE_SQLITE_WAL=false \
    /app/venv/bin/open-webui serve --host 127.0.0.1 --port 3000 > /tmp/e2e/logs/openwebui.log 2>&1 & )
for i in $(seq 1 300); do
  a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health)
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/health)
  [ "$a$b$c" = "200200200" ] && break
  sleep 2
done
echo "E2E-HEALTH stub=$a compactor=$b openwebui=$c after ~$((i*2))s; OpenWebUI $(/app/venv/bin/pip show open-webui 2>/dev/null | grep ^Version)"
cd /app
for ph in current grammar; do
  env DATA_DIR=/tmp/e2e/openwebui WEBUI_SECRET_KEY=wb-e2e-secret /app/venv/bin/python /wb/e2e/driver.py $ph "$MODEL" /priv/value.txt 2>&1 < /dev/null \
    | grep -E "^E2E-|Error|error" | tail -8
done
sleep 1
while read -r line; do echo "E2E-STUB $line"; done < /tmp/e2e/stub.jsonl
