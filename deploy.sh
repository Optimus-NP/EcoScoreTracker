#!/bin/bash

# Sustainability Analytics Platform - Vercel Deployment Script

echo "🌱 Sustainability Analytics Platform - Vercel Deployment"
echo "======================================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "🔐 Checking Vercel authentication..."
vercel whoami || vercel login

# Set environment variables
echo "🔧 Setting up environment variables..."
echo "Please set your SESSION_SECRET when prompted:"
vercel env add SESSION_SECRET production

# Deploy to production
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "Your Sustainability Analytics Platform is now live on Vercel."
echo ""
echo "📊 Features included:"
echo "  • Packaging suggestion ML model"
echo "  • Carbon footprint prediction"
echo "  • Product recommendation engine"
echo "  • ESG score analysis"
echo "  • Batch processing capabilities"
echo "  • Interactive dashboard with charts"
echo ""
echo "🎯 Next steps:"
echo "  1. Test all ML models on your live site"
echo "  2. Upload sample data for batch processing"
echo "  3. Configure any additional environment variables"