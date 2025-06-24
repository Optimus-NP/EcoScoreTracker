#!/usr/bin/env python3
"""
Package the Sustainability Analytics Platform for GitHub/Vercel deployment
Creates a clean deployment package with all necessary files
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_deployment_package():
    """Create a deployment-ready package"""
    
    # Create deployment directory
    deploy_dir = "deployment_package"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Files and directories to include
    include_files = [
        "app.py",
        "main.py", 
        "vercel.json",
        "requirements.txt",
        ".gitignore",
        ".vercelignore",
        "README.md",
        "DEPLOYMENT.md"
    ]
    
    include_dirs = [
        "models",
        "routes", 
        "templates",
        "static",
        "utils",
        "data"
    ]
    
    # Copy files
    for file in include_files:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(deploy_dir, file))
            print(f"Copied {file}")
    
    # Copy directories
    for dir_name in include_dirs:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(deploy_dir, dir_name))
            print(f"Copied directory {dir_name}")
    
    # Create zip file
    zip_path = "sustainability_analytics_deployment.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if not file.endswith(('.pyc', '.pyo')):
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, deploy_dir)
                    zipf.write(file_path, arc_path)
    
    print(f"\nDeployment package created: {zip_path}")
    print(f"Deployment directory: {deploy_dir}")
    
    # Create deployment instructions
    with open("DEPLOY_INSTRUCTIONS.txt", "w") as f:
        f.write("""
SUSTAINABILITY ANALYTICS PLATFORM - DEPLOYMENT INSTRUCTIONS
===========================================================

1. UPLOAD TO GITHUB:
   - Go to: https://github.com/Priyanshu2631/Sustainable-Recommendations
   - Upload all files from the 'deployment_package' folder
   - Or extract 'sustainability_analytics_deployment.zip' and upload contents

2. CONNECT TO VERCEL:
   - Go to: https://vercel.com/priyanshu2631s-projects/sustainable-shopping
   - Settings > Git Repository > Connect Repository
   - Select: Priyanshu2631/Sustainable-Recommendations

3. CONFIGURE ENVIRONMENT:
   - In Vercel dashboard: Settings > Environment Variables
   - Add: SESSION_SECRET = (any random string like: sk_12345_abcdef)

4. DEPLOY:
   - Push to GitHub main branch
   - Vercel will auto-deploy
   - Or manually trigger from Vercel dashboard

5. TEST:
   - Visit your deployed URL
   - Test all 4 ML models (packaging, carbon footprint, products, ESG)
   - Try batch CSV upload
   - Verify dashboard charts work

SUPPORT:
- All files are optimized for Vercel serverless deployment
- 60-second timeout configured for ML operations
- Bootstrap dark theme with responsive design
- Chart.js configured for modern browsers

MODELS INCLUDED:
✓ Packaging Suggestion Model
✓ Carbon Footprint Prediction
✓ Product Recommendation Engine  
✓ ESG Score Analysis
✓ Batch Processing System
✓ Interactive Dashboard
""")
    
    print("Created DEPLOY_INSTRUCTIONS.txt")
    
    # Clean up deployment directory
    shutil.rmtree(deploy_dir)
    print(f"Cleaned up {deploy_dir}")

if __name__ == "__main__":
    create_deployment_package()