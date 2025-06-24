# Vercel Deployment Instructions

## Step 1: Push to GitHub

1. Go to your GitHub repository: https://github.com/Priyanshu2631/Sustainable-Recommendations
2. Upload all project files to the repository (or use git commands if you have git access)

## Step 2: Connect to Vercel

1. Go to your Vercel project: https://vercel.com/priyanshu2631s-projects/sustainable-shopping
2. Connect the GitHub repository
3. Set the root directory to `/` (default)
4. Vercel will automatically detect the Flask application

## Step 3: Configure Environment Variables

In your Vercel dashboard, add these environment variables:

- `SESSION_SECRET`: Generate a random string (e.g., `your-secret-key-here`)

## Step 4: Deploy

1. Trigger deployment from Vercel dashboard
2. Or push changes to GitHub (auto-deployment)

## Important Files for Deployment

- `vercel.json` - Vercel configuration
- `requirements.txt` - Python dependencies
- `app.py` - Main Flask application
- `.vercelignore` - Files to exclude from deployment

## Post-Deployment

1. Visit your deployed URL
2. Test all ML models (packaging, carbon footprint, product recommendations, ESG scores)
3. Try batch processing with CSV upload
4. Verify dashboard charts are working

## Troubleshooting

If deployment fails:
1. Check Vercel build logs
2. Verify all dependencies are in requirements.txt
3. Ensure no syntax errors in Python files
4. Check that all import paths are correct