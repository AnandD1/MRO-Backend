from flask import Blueprint, jsonify, request, session
import logging
from db import UserManager

bp_auth = Blueprint("auth", __name__, url_prefix="/api/auth")

# Initialize user manager
user_manager = UserManager()

logger = logging.getLogger(__name__)

# Simple in-memory session store (import from app)
def get_logged_in_users():
    from app import logged_in_users
    return logged_in_users

@bp_auth.route('/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        email = data.get('email', '').strip()
        
        if not username or not password:
            return jsonify({"success": False, "message": "Username and password required"}), 400
        
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
        
        # Create user
        result = user_manager.create_user(username, password, email)
        
        if result["success"]:
            logger.info(f"New user registered: {username}")
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_auth.route('/login', methods=['POST'])
def login():
    """Simple login endpoint for Unity client"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({"success": False, "message": "Username and password required"}), 400
        
        # Authenticate user
        auth_result = user_manager.authenticate_user(username, password)
        
        if not auth_result["success"]:
            return jsonify(auth_result), 401
        
        # Simple session - store user info and mark as logged in
        user_info = auth_result["user"]
        session['user_id'] = user_info["id"]
        session['username'] = user_info["username"]
        session['logged_in'] = True
        
        # Add to simple logged-in users set
        logged_in_users = get_logged_in_users()
        logged_in_users.add(user_info["id"])
        
        logger.info(f"User {username} logged in successfully")
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": user_info,
            "logged_in": True
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_auth.route('/logout', methods=['POST'])
def logout():
    """Simple logout endpoint"""
    try:
        user_id = session.get('user_id')
        username = session.get('username')
        
        if user_id:
            # Remove from logged-in users
            logged_in_users = get_logged_in_users()
            logged_in_users.discard(user_id)
            logger.info(f"User {username} logged out successfully")
        
        # Clear session
        session.clear()
        
        return jsonify({
            "success": True,
            "message": "Logged out successfully",
            "logged_in": False
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_auth.route('/profile', methods=['GET'])
def get_profile():
    """Get current user profile (simple session check)"""
    try:
        if not session.get('logged_in') or not session.get('user_id'):
            return jsonify({"success": False, "message": "Not logged in"}), 401
        
        user_info = user_manager.get_user_by_id(session.get('user_id'))
        
        if not user_info:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        return jsonify({
            "success": True,
            "user": user_info,
            "logged_in": True
        }), 200
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_auth.route('/check_login', methods=['GET'])
def check_login():
    """Check if user is logged in (simple boolean check)"""
    try:
        is_logged_in = bool(session.get('logged_in') and session.get('user_id'))
        user_id = session.get('user_id')
        
        # Double check with in-memory store
        logged_in_users = get_logged_in_users()
        is_logged_in = is_logged_in and user_id in logged_in_users
        
        if is_logged_in:
            user_info = user_manager.get_user_by_id(user_id)
            return jsonify({
                "success": True,
                "logged_in": True,
                "user": user_info
            }), 200
        else:
            return jsonify({
                "success": True,
                "logged_in": False,
                "user": None
            }), 200
        
    except Exception as e:
        logger.error(f"Login check error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
