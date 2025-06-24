from flask import Blueprint, request, jsonify
from models.packaging_model import packaging_model
from models.carbon_footprint_model import carbon_footprint_model
from models.product_recommendation_model import product_recommendation_model
from models.esg_score_model import esg_score_model
from utils.data_processor import process_batch_data
import logging

api_bp = Blueprint('api', __name__)

@api_bp.route('/packaging/predict', methods=['POST'])
def predict_packaging():
    """API endpoint for packaging prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Material_Type', 'Product_Weight_g', 'Fragility', 'Recyclable', 'Transport_Mode', 'LCA_Emission_kgCO2']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = packaging_model.predict(data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in packaging prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/packaging/feature-importance', methods=['GET'])
def packaging_feature_importance():
    """API endpoint for packaging model feature importance"""
    try:
        result = packaging_model.get_feature_importance()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting packaging feature importance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/carbon-footprint/predict', methods=['POST'])
def predict_carbon_footprint():
    """API endpoint for carbon footprint prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Total_Purchases', 'Avg_Distance_km', 'Preferred_Packaging', 'Returns_%', 'Electricity_kWh', 'Travel_km', 'Service_Usage_hr']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = carbon_footprint_model.predict(data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in carbon footprint prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/carbon-footprint/feature-importance', methods=['GET'])
def carbon_footprint_feature_importance():
    """API endpoint for carbon footprint model feature importance"""
    try:
        result = carbon_footprint_model.get_feature_importance()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting carbon footprint feature importance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/product-recommendation/predict', methods=['POST'])
def predict_product_recommendation():
    """API endpoint for product recommendation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['category', 'material', 'brand', 'price', 'rating', 'reviewsCount', 'Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG', 'Average_Price_USD']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = product_recommendation_model.predict(data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in product recommendation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/product-recommendation/feature-importance', methods=['GET'])
def product_recommendation_feature_importance():
    """API endpoint for product recommendation model feature importance"""
    try:
        result = product_recommendation_model.get_feature_importance()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting product recommendation feature importance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/esg-score/predict', methods=['POST'])
def predict_esg_score():
    """API endpoint for ESG score prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Product Name', 'Sentence', 'Sentiment', 'Environmental Score']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = esg_score_model.predict(data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in ESG score prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/esg-score/feature-importance', methods=['GET'])
def esg_score_feature_importance():
    """API endpoint for ESG score model feature importance"""
    try:
        result = esg_score_model.get_feature_importance()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting ESG score feature importance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/batch-process', methods=['POST'])
def batch_process():
    """API endpoint for batch processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        model_type = request.form.get('model_type')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not model_type:
            return jsonify({'success': False, 'error': 'Model type not specified'}), 400
        
        result = process_batch_data(file, model_type)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/models/status', methods=['GET'])
def models_status():
    """API endpoint to get status of all models"""
    try:
        status = {
            'packaging': packaging_model.is_trained,
            'carbon_footprint': carbon_footprint_model.is_trained,
            'product_recommendation': product_recommendation_model.is_trained,
            'esg_score': esg_score_model.is_trained
        }
        return jsonify({'success': True, 'models_status': status})
    except Exception as e:
        logging.error(f"Error getting models status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/models/train', methods=['POST'])
def train_models():
    """API endpoint to train all models"""
    try:
        results = {}
        
        # Train packaging model
        results['packaging'] = packaging_model.train()
        
        # Train carbon footprint model
        results['carbon_footprint'] = carbon_footprint_model.train()
        
        # Train product recommendation model
        results['product_recommendation'] = product_recommendation_model.train()
        
        # Train ESG score model
        results['esg_score'] = esg_score_model.train()
        
        return jsonify({'success': True, 'training_results': results})
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
