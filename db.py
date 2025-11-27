import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# MongoDB connection
def get_db():
    """Get MongoDB database connection"""
    try:
        # Get connection string from environment (with your correct URI as fallback)
        mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://anand2004gd_db_user:Q5SefnqLkapKfQwj@ai-mro.ylcrzs8.mongodb.net/AI-MRO?retryWrites=true&w=majority')
        client = MongoClient(mongo_uri)
        
        # Use AI-MRO database (matches your cluster and database name)
        db = client['AI-MRO']
        
        # Test the connection
        client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB: {db.name}")
        
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

# User management functions
class UserManager:
    def __init__(self):
        self.db = get_db()
        self.users = self.db.users
        
    def create_user(self, username, password, email=None, role="user"):
        """Create a new user"""
        try:
            # Check if user already exists
            if self.users.find_one({"username": username}):
                return {"success": False, "message": "Username already exists"}
            
            # Hash password
            password_hash = generate_password_hash(password)
            
            # Create user document
            user_doc = {
                "username": username,
                "password_hash": password_hash,
                "email": email,
                "role": role,
                "created_at": datetime.utcnow(),
                "last_login": None,
                "is_active": True
            }
            
            # Insert user
            result = self.users.insert_one(user_doc)
            
            if result.inserted_id:
                return {"success": True, "message": "User created successfully", "user_id": str(result.inserted_id)}
            else:
                return {"success": False, "message": "Failed to create user"}
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {"success": False, "message": "Database error"}
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials"""
        try:
            user = self.users.find_one({"username": username, "is_active": True})
            
            if not user:
                return {"success": False, "message": "Invalid username or password"}
            
            if check_password_hash(user["password_hash"], password):
                # Update last login
                self.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )
                
                # Return user info (without password)
                user_info = {
                    "id": str(user["_id"]),
                    "username": user["username"],
                    "email": user.get("email"),
                    "role": user.get("role", "user"),
                    "last_login": user.get("last_login")
                }
                
                return {"success": True, "user": user_info}
            else:
                return {"success": False, "message": "Invalid username or password"}
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {"success": False, "message": "Database error"}
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            from bson import ObjectId
            user = self.users.find_one({"_id": ObjectId(user_id), "is_active": True})
            
            if user:
                return {
                    "id": str(user["_id"]),
                    "username": user["username"],
                    "email": user.get("email"),
                    "role": user.get("role", "user"),
                    "last_login": user.get("last_login")
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

# Scan management functions
class ScanManager:
    def __init__(self):
        self.db = get_db()
        self.scans = self.db.scans
        
    def create_scan(self, user_id, scan_data, scan_name=None):
        """Create a new scan for a user"""
        try:
            from bson import ObjectId
            
            # Validate required fields in scan_data
            if not scan_data.get("version"):
                return {"success": False, "message": "Missing version field"}
            if not scan_data.get("plane"):
                return {"success": False, "message": "Missing plane data"}
            if not scan_data.get("markers"):
                return {"success": False, "message": "Missing markers array"}
            
            # Determine scan type based on markers
            markers = scan_data.get("markers", [])
            has_severity = any(m.get("severity") for m in markers)
            scan_type = "manual" if has_severity else "ai"
            
            # Generate scan name if not provided
            if not scan_name:
                scan_name = f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create scan document
            scan_doc = {
                "user_id": ObjectId(user_id),
                "scan_name": scan_name,
                "scan_type": scan_type,
                "version": scan_data.get("version"),
                "ts_ms": scan_data.get("ts_ms"),
                "plane": scan_data.get("plane"),
                "markers": markers,
                "marker_count": len(markers),
                "created_at": datetime.utcnow(),
                "metadata": scan_data.get("metadata", {})
            }
            
            # Insert scan
            result = self.scans.insert_one(scan_doc)
            
            if result.inserted_id:
                return {
                    "success": True,
                    "message": "Scan created successfully",
                    "scan_id": str(result.inserted_id),
                    "scan_type": scan_type,
                    "marker_count": len(markers)
                }
            else:
                return {"success": False, "message": "Failed to create scan"}
                
        except Exception as e:
            logger.error(f"Error creating scan: {e}")
            return {"success": False, "message": "Database error"}
    
    def get_all_scans(self, limit=100, skip=0):
        """Get all scans (globally accessible) with pagination"""
        try:
            cursor = self.scans.find(
                {},  # No user filter - get all scans
                {"markers": 0}  # Exclude markers array for list view
            ).sort("created_at", -1).skip(skip).limit(limit)
            
            scans = []
            for scan in cursor:
                scans.append({
                    "id": str(scan["_id"]),
                    "scan_name": scan.get("scan_name"),
                    "scan_type": scan.get("scan_type"),
                    "marker_count": scan.get("marker_count", 0),
                    "created_at": scan.get("created_at").isoformat() + "Z" if scan.get("created_at") else None,
                    "ts_ms": scan.get("ts_ms"),
                    "created_by": str(scan.get("user_id"))  # Include creator info
                })
            
            return {"success": True, "scans": scans, "count": len(scans)}
            
        except Exception as e:
            logger.error(f"Error getting all scans: {e}")
            return {"success": False, "message": "Database error"}
    
    def get_scan_by_id(self, scan_id):
        """Get a specific scan by ID (globally accessible)"""
        try:
            from bson import ObjectId
            
            scan = self.scans.find_one({"_id": ObjectId(scan_id)})  # No user filter
            
            if not scan:
                return {"success": False, "message": "Scan not found"}
            
            return {
                "success": True,
                "scan": {
                    "id": str(scan["_id"]),
                    "scan_name": scan.get("scan_name"),
                    "scan_type": scan.get("scan_type"),
                    "version": scan.get("version"),
                    "ts_ms": scan.get("ts_ms"),
                    "plane": scan.get("plane"),
                    "markers": scan.get("markers", []),
                    "marker_count": scan.get("marker_count", 0),
                    "created_at": scan.get("created_at").isoformat() + "Z" if scan.get("created_at") else None,
                    "created_by": str(scan.get("user_id")),  # Include creator info
                    "metadata": scan.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting scan: {e}")
            return {"success": False, "message": "Database error"}
    
    def delete_scan(self, scan_id):
        """Delete a scan (globally accessible - any authenticated user can delete)"""
        try:
            from bson import ObjectId
            
            result = self.scans.delete_one({"_id": ObjectId(scan_id)})  # No user filter
            
            if result.deleted_count > 0:
                return {"success": True, "message": "Scan deleted successfully"}
            else:
                return {"success": False, "message": "Scan not found"}
                
        except Exception as e:
            logger.error(f"Error deleting scan: {e}")
            return {"success": False, "message": "Database error"}
