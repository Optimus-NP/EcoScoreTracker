import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging

class ProductRecommendationModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_names = ['category', 'material', 'brand', 'price', 'rating', 'reviewsCount', 'Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG', 'Average_Price_USD']
        self.categorical_cols = ['category', 'material', 'brand']
        
    def create_sample_data(self):
        """Create sample training data for the product recommendation model"""
        np.random.seed(42)
        n_samples = 1000
        
        categories = ['Clothing', 'Electronics', 'Beauty', 'Home', 'Sports']
        materials = ['Cotton', 'Polyester', 'Metal', 'Plastic', 'Wood', 'Glass']
        brands = ['EcoBrand', 'GreenTech', 'SustainableCorp', 'NaturalGoods', 'CleanLiving']
        
        data = []
        for i in range(n_samples):
            category = np.random.choice(categories)
            material = np.random.choice(materials)
            brand = np.random.choice(brands)
            
            # Generate correlated features
            if category == 'Electronics':
                price_base = 200
                carbon_base = 15
                water_base = 500
                waste_base = 8
            elif category == 'Clothing':
                price_base = 50
                carbon_base = 8
                water_base = 2000
                waste_base = 2
            elif category == 'Beauty':
                price_base = 30
                carbon_base = 3
                water_base = 100
                waste_base = 1
            elif category == 'Home':
                price_base = 100
                carbon_base = 12
                water_base = 300
                waste_base = 5
            else:  # Sports
                price_base = 75
                carbon_base = 6
                water_base = 800
                waste_base = 3
            
            # Add randomness
            price = max(10, np.random.normal(price_base, price_base * 0.3))
            carbon_footprint = max(1, np.random.normal(carbon_base, carbon_base * 0.4))
            water_usage = max(50, np.random.normal(water_base, water_base * 0.3))
            waste_production = max(0.5, np.random.normal(waste_base, waste_base * 0.4))
            average_price_usd = price * np.random.uniform(0.8, 1.2)
            
            # Rating and review count (correlated with sustainability)
            sustainability_score = (
                (20 - carbon_footprint) / 20 * 0.4 +  # Lower carbon is better
                (3000 - water_usage) / 3000 * 0.3 +   # Lower water usage is better
                (10 - waste_production) / 10 * 0.3    # Lower waste is better
            )
            sustainability_score = max(0, min(1, sustainability_score))
            
            rating = 2 + sustainability_score * 3 + np.random.normal(0, 0.3)
            rating = max(1, min(5, rating))
            
            reviews_count = int(50 + sustainability_score * 400 + np.random.exponential(100))
            
            # Purchase likelihood based on rating and reviews
            purchase_prob = (rating - 1) / 4 * 0.6 + min(reviews_count / 500, 1) * 0.4
            purchased = np.random.random() < purchase_prob
            
            data.append({
                'product_id': f'P{i+1:04d}',
                'category': category,
                'material': material,
                'brand': brand,
                'price': round(price, 2),
                'rating': round(rating, 1),
                'reviewsCount': reviews_count,
                'Carbon_Footprint_MT': round(carbon_footprint, 2),
                'Water_Usage_Liters': round(water_usage, 1),
                'Waste_Production_KG': round(waste_production, 2),
                'Average_Price_USD': round(average_price_usd, 2),
                'purchased': int(purchased)
            })
        
        return pd.DataFrame(data)
    
    def train(self, df=None):
        """Train the product recommendation model"""
        try:
            if df is None:
                df = self.create_sample_data()
                
            logging.info("Training product recommendation model...")
            
            # Encode categoricals
            for col in self.categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            
            # Prepare feature matrix and target vector
            X = df[self.feature_names]
            y = df['purchased']
            
            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
            
            # Grid search
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            
            return {
                'success': True,
                'accuracy': accuracy,
                'best_params': grid_search.best_params_
            }
            
        except Exception as e:
            logging.error(f"Error training product recommendation model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, input_data):
        """Make product recommendation prediction"""
        try:
            if not self.is_trained:
                train_result = self.train()
                if not train_result['success']:
                    return {'success': False, 'error': 'Failed to train model'}
            
            # Prepare input data
            df_input = pd.DataFrame([input_data])
            
            # Encode categorical features
            for col in self.categorical_cols:
                if col in df_input.columns:
                    if col in self.label_encoders:
                        try:
                            df_input[col] = self.label_encoders[col].transform([input_data[col]])[0]
                        except ValueError:
                            # If category not seen during training, use most common
                            df_input[col] = 0
            
            # Scale features
            df_scaled = self.scaler.transform(df_input)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(df_scaled)[0]
            purchase_likelihood = float(prediction_proba[1])  # Probability of purchase
            
            return {
                'success': True,
                'purchase_likelihood': purchase_likelihood,
                'recommendation': purchase_likelihood > 0.5
            }
            
        except Exception as e:
            logging.error(f"Error making product recommendation: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self):
        """Get feature importance for the trained model"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            importances = self.model.feature_importances_
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, importances)
            }
            
            return {'success': True, 'feature_importance': feature_importance}
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global instance
product_recommendation_model = ProductRecommendationModel()
