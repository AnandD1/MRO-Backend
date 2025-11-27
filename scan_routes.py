from flask import Blueprint, jsonify, request, session
import logging
from db import ScanManager
from utils import session_required

bp_scan = Blueprint("scans", __name__, url_prefix="/api/scans")

# Initialize scan manager
scan_manager = ScanManager()

logger = logging.getLogger(__name__)

@bp_scan.route('/', methods=['POST'])
@bp_scan.route('', methods=['POST'])
@session_required
def create_scan():
    """
    Create a new scan from Unity client.
    Expects JSON body matching the scan format:
    {
      "version": "1.0",
      "ts_ms": <timestamp>,
      "plane": { ... },
      "markers": [ ... ]
    }
    Optional query parameter: ?name=custom_scan_name
    """
    try:
        data = request.get_json(silent=True)
        
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided"}), 400
        
        # Get user_id from session
        user_id = session.get('user_id')
        
        # Get optional scan name from query params or request body
        scan_name = request.args.get('name') or data.get('scan_name')
        
        # Create scan
        result = scan_manager.create_scan(user_id, data, scan_name)
        
        if result["success"]:
            logger.info(f"Scan created for user {user_id}: {result.get('scan_id')}")
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error creating scan: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_scan.route('/', methods=['GET'])
@bp_scan.route('', methods=['GET'])
@session_required
def list_scans():
    """
    List all scans globally (accessible to all authenticated users).
    Optional query parameters:
    - limit: number of scans to return (default: 100)
    - skip: number of scans to skip for pagination (default: 0)
    """
    try:
        # Get pagination parameters
        limit = request.args.get('limit', 100, type=int)
        skip = request.args.get('skip', 0, type=int)
        
        # Validate pagination
        limit = min(max(1, limit), 500)  # Clamp between 1 and 500
        skip = max(0, skip)
        
        # Get all scans (no user filtering)
        result = scan_manager.get_all_scans(limit, skip)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error listing scans: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_scan.route('/<scan_id>', methods=['GET'])
@session_required
def get_scan(scan_id):
    """
    Get a specific scan by ID (globally accessible to all authenticated users).
    """
    try:
        # Get scan without user filtering
        result = scan_manager.get_scan_by_id(scan_id)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"Error getting scan: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@bp_scan.route('/<scan_id>', methods=['DELETE'])
@session_required
def delete_scan(scan_id):
    """
    Delete a specific scan by ID (globally accessible - any authenticated user can delete).
    """
    try:
        # Delete scan without user filtering
        result = scan_manager.delete_scan(scan_id)
        
        if result["success"]:
            logger.info(f"Scan {scan_id} deleted")
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"Error deleting scan: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
