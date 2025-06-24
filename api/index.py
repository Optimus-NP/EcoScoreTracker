import os
import sys
from flask import Flask

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app from the parent directory
try:
    from app import app
except ImportError:
    # Fallback if import fails
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Vercel expects the app to be available at module level
# Export the Flask app instance
def create_app():
    return app

# For Vercel deployment
application = app

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))