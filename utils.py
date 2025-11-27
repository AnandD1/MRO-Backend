from flask import request, jsonify, session
from functools import wraps

def json_required(*fields):
    """Helper to validate required JSON fields"""
    data = request.get_json(silent=True) or {}
    missing = [f for f in fields if f not in data or data[f] in (None, "")]
    if missing:
        return None, jsonify({"success": False, "error": "missing_fields", "fields": missing}), 400
    return data, None, None

def validate_email(email):
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def session_required(f):
    """Decorator to require valid session authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
            return jsonify({"success": False, "error": "authentication_required", "message": "Please log in"}), 401
        return f(*args, **kwargs)
    return decorated_function
