import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

class CarbonFootprintModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        self.feature_names = ['Total_Purchases', 'Avg_Distance_km', 'Preferred_Packaging', 'Returns_%', 'Electricity_kWh', 'Travel_km', 'Service_Usage_hr']
        
    def create_sample_data(self):
        """Create sample training data for the carbon footprint model"""
        np.random.seed(42)
        n_samples = 1000
        
        packaging_options = ['Cardboard', 'Plastic', 'Paper', 'Metal', 'Glass']
        
        data = []
        for i in range(n_samples):
            # Generate correlated features
            total_purchases = np.random.poisson(20)
            avg_distance = np.random.exponential(300)
            preferred_packaging = np.random.choice(packaging_options)
            returns_pct = np.random.beta(2, 8) * 15  # Mostly low returns
            electricity = np.random.normal(350, 100)
            travel_km = np.random.exponential(800)
            service_usage = np.random.gamma(3, 8)
            
            # Calculate CO2 emissions based on features
            co2_base = (
                total_purchases * 2.5 +  # Each purchase contributes to CO2
                avg_distance * 0.2 +     # Distance affects shipping emissions
                returns_pct * 5 +        # Returns increase emissions
                electricity * 0.5 +      # Electricity usage
                travel_km * 0.3 +        # Personal travel
                service_usage * 1.2      # Service usage
            )
            
            # Packaging multiplier
            packaging_multipliers = {
                'Cardboard': 0.8,  # More eco-friendly
                'Paper': 0.9,
                'Metal': 1.1,
                'Plastic': 1.3,    # Less eco-friendly
                'Glass': 1.2
            }
            
            co2_emission = co2_base * packaging_multipliers[preferred_packaging]
            co2_emission += np.random.normal(0, 50)  # Add noise
            co2_emission = max(50, co2_emission)     # Minimum emissions
            
            data.append({
                'Customer_ID': f'C{i+1:04d}',
                'Total_Purchases': int(total_purchases),
                'Avg_Distance_km': round(avg_distance, 1),
                'Preferred_Packaging': preferred_packaging,
                'Returns_%': round(returns_pct, 1),
                'Electricity_kWh': round(electricity, 1),
                'Travel_km': round(travel_km, 1),
                'Service_Usage_hr': round(service_usage, 1),
                'Est_CO2e_kg': round(co2_emission, 2)
            })
        
        return pd.DataFrame(data)
    
    def train(self, df=None):
        """Train the carbon footprint prediction model"""
        try:
            if df is None:
                df = self.create_sample_data()
                
            logging.info("Training carbon footprint model...")
            
            # Encode categorical variable
            self.label_encoder = LabelEncoder()
            df['Preferred_Packaging'] = self.label_encoder.fit_transform(df['Preferred_Packaging'])
            
            # Features and target
            X = df.drop(columns=['Customer_ID', 'Est_CO2e_kg'])
            y = df['Est_CO2e_kg']
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(random_state=42))
            ])
            
            # Define hyperparameter grid
            param_grid = {
                'model__n_estimators': [100, 150],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 4],
                'model__subsample': [0.8, 1.0]
            }
            
            # Run GridSearchCV
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Evaluate performance
            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model trained successfully. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            
            return {
                'success': True,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'best_params': grid_search.best_params_
            }
            
        except Exception as e:
            logging.error(f"Error training carbon footprint model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, input_data):
        """Make carbon footprint prediction"""
        try:
            if not self.is_trained:
                train_result = self.train()
                if not train_result['success']:
                    return {'success': False, 'error': 'Failed to train model'}
            
            # Prepare input data
            df_input = pd.DataFrame([input_data])
            
            # Encode preferred packaging
            if 'Preferred_Packaging' in df_input.columns:
                try:
                    df_input['Preferred_Packaging'] = self.label_encoder.transform([input_data['Preferred_Packaging']])[0]
                except ValueError:
                    # If category not seen during training, use most common
                    df_input['Preferred_Packaging'] = 0
            
            # Make prediction
            prediction = self.model.predict(df_input)[0]
            
            return {
                'success': True,
                'prediction': float(prediction),
                'prediction_rounded': round(prediction)
            }
            
        except Exception as e:
            logging.error(f"Error making carbon footprint prediction: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self):
        """Get feature importance for the trained model"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Get feature importance from the model step of the pipeline
            model_step = self.model.named_steps['model']
            importances = model_step.feature_importances_
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, importances)
            }
            
            return {'success': True, 'feature_importance': feature_importance}
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global instance
carbon_footprint_model = CarbonFootprintModel()
