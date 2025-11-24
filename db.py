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
