"""Trivial appservice transaction receiver.

Synapse will PUT /transactions/{txnId} to the appservice's registered `url`
whenever it wants to push events to it. Our importer is outbound-only (it
sends historical events, it doesn't need Synapse to relay anything back), so
this just acknowledges every transaction with an empty 200 so Synapse's
retry/backoff queue never backs up. It also answers the AS user/room "exists"
query endpoints Synapse may probe.
"""
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/transactions/<txn_id>", methods=["PUT"])
def transactions(txn_id):
    return jsonify({})


@app.route("/users/<user_id>", methods=["GET"])
def users(user_id):
    return jsonify({})


@app.route("/rooms/<alias>", methods=["GET"])
def rooms(alias):
    return jsonify({}), 404


@app.route("/_matrix/app/v1/transactions/<txn_id>", methods=["PUT"])
def transactions_v1(txn_id):
    return jsonify({})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9200)
