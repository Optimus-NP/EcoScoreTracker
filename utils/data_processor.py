import pandas as pd
import io
import logging
from models.packaging_model import packaging_model
from models.carbon_footprint_model import carbon_footprint_model
from models.product_recommendation_model import product_recommendation_model
from models.esg_score_model import esg_score_model

def process_batch_data(file, model_type):
    """Process batch data for predictions"""
    try:
        # Read the uploaded file
        file_content = file.read()
        
        # Try to read as CSV
        try:
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        except Exception as e:
            return {'success': False, 'error': f'Error reading CSV file: {str(e)}'}
        
        if df.empty:
            return {'success': False, 'error': 'Uploaded file is empty'}
        
        results = []
        errors = []
        
        # Process each row based on model type
        for index, row in df.iterrows():
            try:
                row_data = row.to_dict()
                
                if model_type == 'packaging':
                    result = packaging_model.predict(row_data)
                elif model_type == 'carbon_footprint':
                    result = carbon_footprint_model.predict(row_data)
                elif model_type == 'product_recommendation':
                    result = product_recommendation_model.predict(row_data)
                elif model_type == 'esg_score':
                    result = esg_score_model.predict(row_data)
                else:
                    return {'success': False, 'error': f'Unknown model type: {model_type}'}
                
                if result['success']:
                    row_result = row_data.copy()
                    row_result.update(result)
                    row_result['row_index'] = index
                    results.append(row_result)
                else:
                    errors.append({
                        'row_index': index,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                errors.append({
                    'row_index': index,
                    'error': str(e)
                })
        
        return {
            'success': True,
            'total_rows': len(df),
            'successful_predictions': len(results),
            'failed_predictions': len(errors),
            'results': results,
            'errors': errors
        }
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")
        return {'success': False, 'error': str(e)}

def validate_input_data(data, model_type):
    """Validate input data for a specific model type"""
    required_fields = {
        'packaging': ['Material_Type', 'Product_Weight_g', 'Fragility', 'Recyclable', 'Transport_Mode', 'LCA_Emission_kgCO2'],
        'carbon_footprint': ['Total_Purchases', 'Avg_Distance_km', 'Preferred_Packaging', 'Returns_%', 'Electricity_kWh', 'Travel_km', 'Service_Usage_hr'],
        'product_recommendation': ['category', 'material', 'brand', 'price', 'rating', 'reviewsCount', 'Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG', 'Average_Price_USD'],
        'esg_score': ['Product Name', 'Sentence', 'Sentiment', 'Environmental Score']
    }
    
    if model_type not in required_fields:
        return False, f'Unknown model type: {model_type}'
    
    missing_fields = []
    for field in required_fields[model_type]:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f'Missing required fields: {", ".join(missing_fields)}'
    
    return True, 'Valid'

def format_prediction_result(result, model_type):
    """Format prediction result for display"""
    if not result['success']:
        return result
    
    formatted_result = result.copy()
    
    if model_type == 'packaging':
        formatted_result['display_text'] = f"Recommended packaging: {result['prediction']} (Confidence: {result['confidence']:.2%})"
    elif model_type == 'carbon_footprint':
        formatted_result['display_text'] = f"Estimated carbon footprint: {result['prediction_rounded']} kg CO2e"
    elif model_type == 'product_recommendation':
        recommendation = "Recommended" if result['recommendation'] else "Not Recommended"
        formatted_result['display_text'] = f"{recommendation} (Likelihood: {result['purchase_likelihood']:.2%})"
    elif model_type == 'esg_score':
        formatted_result['display_text'] = f"ESG Score: {result['prediction_rounded']} ({result['status']})"
    
    return formatted_result
