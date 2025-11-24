from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
from db import get_db
from utils import json_required
from bson import ObjectId

bp_scan = Blueprint("scans", __name__, url_prefix="/api/scans")

@bp_scan.post("/")
@bp_scan.post("")   # allow no trailing slash
@jwt_required()
def create_scan():
    """
    Save a scan JSON posted by the headset or a tool.
    Expect the body to match your 'Exported scan' schema.
    Optional: scan_name string for easier listing.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    email = get_jwt_identity()
    db = get_db()
    user = db.users.find_one({"email": email}, {"_id": 1})
    if not user:
        return jsonify({"ok": False, "error": "user_not_found"}), 404

    doc = {
        "user_id": user["_id"],
        "scan_name": request.args.get("name") or f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "payload": data,  # store full JSON payload
        "created_at": datetime.utcnow(),
    }
    ins = db.scans.insert_one(doc)
    return jsonify({"ok": True, "id": str(ins.inserted_id)}), 201

@bp_scan.get("/")
@jwt_required()
def list_scans():
    email = get_jwt_identity()
    db = get_db()
    user = db.users.find_one({"email": email}, {"_id": 1})
    cur = db.scans.find({"user_id": user["_id"]}, {"payload": 0}).sort("created_at", -1)
    out = []
    for s in cur:
        out.append({
            "id": str(s["_id"]),
            "scan_name": s.get("scan_name"),
            "created_at": s.get("created_at").isoformat() + "Z"
        })
    return jsonify({"ok": True, "scans": out})

@bp_scan.get("/<scan_id>")
@jwt_required()
def get_scan(scan_id):
    email = get_jwt_identity()
    db = get_db()
    user = db.users.find_one({"email": email}, {"_id": 1})
    try:
        _id = ObjectId(scan_id)
    except Exception:
        return jsonify({"ok": False, "error": "bad_id"}), 400
    s = db.scans.find_one({"_id": _id, "user_id": user["_id"]})
    if not s:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "scan": {
        "id": str(s["_id"]),
        "scan_name": s.get("scan_name"),
        "created_at": s.get("created_at").isoformat() + "Z",
        "payload": s.get("payload"),
    }})
