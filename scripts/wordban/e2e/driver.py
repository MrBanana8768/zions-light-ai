"""Drive the REAL OpenWebUI (from the image) the way the browser does, for `wordban.py e2e`: the model edit through
/api/v1/models/model/update (what Admin > Models > Save & Update posts), then chats through /api/chat/completions.
Runs with OpenWebUI's own venv. Args: <phase: current|grammar> <model id> <value file>"""
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request

from open_webui.utils.auth import create_token

BASE = "http://127.0.0.1:3000"
phase, MODEL, VALUE = sys.argv[1], sys.argv[2], sys.argv[3]
db = sqlite3.connect(os.environ["DATA_DIR"] + "/webui.db")
uid = db.execute("select id from user where role='admin' order by created_at limit 1").fetchone()[0]
TOKEN = create_token({"id": uid})


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
        try:
            return r.status, json.loads(raw)
        except Exception:                                           # noqa: BLE001
            return r.status, raw


def chat(tag, stream):
    st, resp = call("POST", "/api/chat/completions",
                    {"model": MODEL, "stream": stream,
                     "messages": [{"role": "user", "content": f"[wb-e2e {tag}] Say something."}]})
    print(f"E2E-CHAT {json.dumps({'tag': tag, 'stream': stream, 'http': st})}", flush=True)


def get_model():
    return call("GET", "/api/v1/models/model?id=" + urllib.parse.quote(MODEL, safe=""))[1]


def edit(custom_updates, drop=()):
    m = get_model()
    params = dict(m["params"])
    cp = dict(params.get("custom_params") or {})
    for k in drop:
        cp.pop(k, None)
    cp.update(custom_updates)
    params["custom_params"] = cp
    form = {"id": m["id"], "base_model_id": m.get("base_model_id"), "name": m["name"], "meta": m["meta"],
            "params": params, "is_active": m.get("is_active", True)}
    if "access_grants" in m:
        form["access_grants"] = m["access_grants"]
    st, _ = call("POST", "/api/v1/models/model/update", form)
    saved = (get_model()["params"].get("custom_params") or {}).get("structured_outputs")
    print(f"E2E-EDIT {json.dumps({'http': st, 'saved_chars': len(saved or ''), 'identical_to_file': saved == open(VALUE, encoding='utf-8').read()})}",
          flush=True)


if phase == "current":
    chat("A-current", False)
elif phase == "grammar":
    edit({"structured_outputs": open(VALUE, encoding="utf-8").read()}, drop=("bad_words",))
    chat("B-grammar", False)
    chat("B-grammar-stream", True)
