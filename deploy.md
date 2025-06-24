# Sustainability Analytics Platform - Vercel Deployment Guide

## Prerequisites
- Vercel account with CLI access
- Node.js installed locally (for Vercel CLI)

## Deployment Steps

### 1. Install Vercel CLI
```bash
npm i -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy the Application
```bash
vercel --prod
```

## Configuration Files Created

### vercel.json
- Configures Python runtime for Flask application
- Sets up routing to handle all requests through app.py
- Configures 60-second timeout for ML model operations

### requirements.txt
- Lists all Python dependencies needed for deployment
- Optimized for Vercel's serverless environment

### .vercelignore
- Excludes development files from deployment
- Reduces deployment package size

## Environment Variables

Set these in your Vercel dashboard:

1. **SESSION_SECRET**: Random string for Flask sessions
   ```
   vercel env add SESSION_SECRET
   ```

## Post-Deployment Notes

1. **Model Training**: Models will be trained on first request (cold start)
2. **Static Files**: All static assets (CSS, JS, images) will be served by Vercel
3. **Database**: Currently using in-memory storage - consider adding persistent database for production

## Vercel Limitations to Consider

- **Serverless Functions**: 60-second timeout limit
- **Memory**: 1GB RAM limit for hobby plan
- **Cold Starts**: First request may be slower
- **File System**: Read-only after deployment

## Alternative Deployment Options

If Vercel limitations affect your ML models, consider:
- **Railway**: Better for persistent applications
- **Render**: Good Python support with persistent storage
- **Heroku**: Traditional platform-as-a-service

## Support

The application is optimized for Vercel deployment with:
- Serverless-friendly Flask configuration
- Optimized Chart.js setup for modern browsers
- Bootstrap dark theme integration
- Responsive design for all devices