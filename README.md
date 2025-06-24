# Sustainability Analytics Platform

A comprehensive web application that integrates four machine learning models for sustainability and eco-friendly product analysis.

## Features

- **Packaging Suggestion Model**: Recommends optimal packaging based on product characteristics
- **Carbon Footprint Prediction**: Calculates environmental impact for customer profiles
- **Product Recommendation Engine**: Suggests eco-friendly alternatives with sustainability metrics
- **ESG Score Analysis**: Analyzes Environmental, Social, and Governance scores using sentiment analysis
- **Batch Processing**: Upload CSV files for bulk analysis
- **Interactive Dashboard**: Real-time predictions with feature importance visualization

## Technology Stack

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5 (Dark Theme), Chart.js
- **Deployment**: Vercel

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Vercel Deployment
This application is optimized for Vercel deployment with serverless functions.

## Environment Variables

Set these in your Vercel dashboard:
- `SESSION_SECRET`: Random string for Flask sessions

## Project Structure

```
├── app.py                 # Flask application setup
├── main.py               # Application entry point
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── models/               # ML model implementations
├── routes/               # Flask route handlers
├── templates/            # HTML templates
├── static/               # CSS, JS, and assets
└── utils/                # Data processing utilities
```

## Models

### 1. Packaging Model
Suggests optimal packaging materials based on:
- Product weight and fragility
- Material type and recyclability
- Transportation mode

### 2. Carbon Footprint Model
Calculates environmental impact considering:
- Customer demographics and location
- Product preferences and usage patterns
- Transportation and energy consumption

### 3. Product Recommendation Model
Recommends sustainable alternatives using:
- Product categories and features
- Sustainability ratings
- Customer preferences

### 4. ESG Score Model
Analyzes sustainability metrics through:
- Sentiment analysis of ESG reports
- Environmental impact scoring
- Social and governance factors

## API Endpoints

- `POST /api/predict/packaging` - Get packaging suggestions
- `POST /api/predict/carbon-footprint` - Calculate carbon footprint
- `POST /api/predict/product-recommendation` - Get product recommendations
- `POST /api/predict/esg-score` - Analyze ESG scores
- `POST /api/batch-process` - Process CSV files
- `GET /api/models/status` - Check model training status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details