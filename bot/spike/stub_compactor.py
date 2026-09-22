"""Stand-in for the real compactor (RunPod / vLLM). Records exactly what the
bot sent it -- headers included -- to a JSON-lines file the orchestration
script reads back afterwards, and returns a canned assistant reply so the
bot has something to encrypt and send back to the room.
"""
import json
import os
import time

from flask import Flask, jsonify, request

app = Flask(__name__)
LOG_PATH = os.environ.get("SPIKE_COMPACTOR_LOG", "/work/out/compactor_requests.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


@app.route("/v1/chat", methods=["POST"])
def chat():
    record = {
        "ts": time.time(),
        "headers": {
            "X-Conversation-Id": request.headers.get("X-Conversation-Id"),
            "Authorization": request.headers.get("Authorization"),
        },
        "body": request.get_json(silent=True),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonify(
        {
            "reply": (
                "[stub-compactor reply] I received your message and this stands "
                "in for a real completion."
            )
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)
