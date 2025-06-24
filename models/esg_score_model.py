import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import logging

class ESGScoreModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.sample_data = None
        
    def load_sample_data(self):
        """Load the provided ESG sentiment data"""
        try:
            # Create sample data based on the provided CSV structure
            products = ['Non-renewable Cars', 'Toxic Household', 'Petroleum-based Cars', 'Eco-friendly Clothing', 
                       'Solar Energy', 'Organic Food', 'Recyclable Packaging', 'Green Electronics']
            sentiments = ['Negative', 'Positive', 'Neutral']
            
            np.random.seed(42)
            data = []
            
            for i in range(500):
                product = np.random.choice(products)
                sentiment = np.random.choice(sentiments)
                
                # Generate environmental score based on product type
                if 'Eco-friendly' in product or 'Solar' in product or 'Organic' in product or 'Recyclable' in product or 'Green' in product:
                    env_score = np.random.uniform(70, 95)
                    esg_base = np.random.uniform(75, 95)
                    sentence_templates = [
                        f"The {product} demonstrates excellent sustainability practices and environmental responsibility.",
                        f"This {product} is highly recommended for its eco-friendly approach and positive impact.",
                        f"Outstanding environmental performance makes {product} a top choice for conscious consumers."
                    ]
                else:
                    env_score = np.random.uniform(20, 50)
                    esg_base = np.random.uniform(25, 55)
                    sentence_templates = [
                        f"The {product} is widely recognized for its impact. Unfortunately, it fails to meet expectations and is not recommended.",
                        f"{product} is widely recognized for its negative environmental impact.",
                        f"This {product} raises significant environmental concerns and sustainability issues."
                    ]
                
                sentence = np.random.choice(sentence_templates)
                
                # Adjust ESG score based on sentiment
                if sentiment == 'Positive':
                    esg_score = esg_base + np.random.uniform(5, 15)
                elif sentiment == 'Negative':
                    esg_score = esg_base - np.random.uniform(5, 15)
                else:  # Neutral
                    esg_score = esg_base + np.random.uniform(-5, 5)
                
                esg_score = max(10, min(100, esg_score))
                status = 'Eco-friendly' if esg_score > 60 else 'Non Eco-friendly'
                
                data.append({
                    'Product Name': product,
                    'Sentence': sentence,
                    'Sentiment': sentiment,
                    'ESG Score': round(esg_score),
                    'Environmental Score': round(env_score),
                    'Status': status
                })
            
            self.sample_data = pd.DataFrame(data)
            return self.sample_data
            
        except Exception as e:
            logging.error(f"Error loading sample data: {str(e)}")
            return None
    
    def train(self, df=None):
        """Train the ESG score prediction model"""
        try:
            if df is None:
                df = self.load_sample_data()
                if df is None:
                    return {'success': False, 'error': 'Failed to load training data'}
                
            logging.info("Training ESG score model...")
            
            # Select features and target
            X = df[['Product Name', 'Sentence', 'Sentiment', 'Environmental Score']]
            y = df['ESG Score']
            
            # Define preprocessor
            preprocessor = ColumnTransformer(transformers=[
                ('product_name', OneHotEncoder(handle_unknown='ignore'), ['Product Name']),
                ('sentiment', OneHotEncoder(handle_unknown='ignore'), ['Sentiment']),
                ('sentence_tfidf', TfidfVectorizer(max_features=300), 'Sentence'),
            ], remainder='passthrough')  # Keeps 'Environmental Score' as is
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Full pipeline
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('regressor', GradientBoostingRegressor(random_state=42))
            ])
            
            # Hyperparameter grid
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [3, 5],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__subsample': [0.8, 1.0]
            }
            
            # Grid search
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2')
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            logging.info(f"Model trained successfully. R2: {r2:.4f}, RMSE: {rmse:.2f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            
            return {
                'success': True,
                'r2_score': r2,
                'rmse': rmse,
                'best_params': grid_search.best_params_
            }
            
        except Exception as e:
            logging.error(f"Error training ESG score model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, input_data):
        """Make ESG score prediction"""
        try:
            if not self.is_trained:
                train_result = self.train()
                if not train_result['success']:
                    return {'success': False, 'error': 'Failed to train model'}
            
            # Prepare input data
            df_input = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(df_input)[0]
            
            # Determine status based on score
            status = 'Eco-friendly' if prediction > 60 else 'Non Eco-friendly'
            
            return {
                'success': True,
                'prediction': float(prediction),
                'prediction_rounded': round(prediction),
                'status': status
            }
            
        except Exception as e:
            logging.error(f"Error making ESG score prediction: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self):
        """Get top feature importance for the trained model"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Get feature names from preprocessor
            preprocessor = self.model.named_steps['preprocessing']
            
            feature_names = []
            
            # Product name features
            if hasattr(preprocessor.named_transformers_['product_name'], 'get_feature_names_out'):
                product_features = preprocessor.named_transformers_['product_name'].get_feature_names_out(['Product Name'])
                feature_names.extend(product_features)
            
            # Sentiment features
            if hasattr(preprocessor.named_transformers_['sentiment'], 'get_feature_names_out'):
                sentiment_features = preprocessor.named_transformers_['sentiment'].get_feature_names_out(['Sentiment'])
                feature_names.extend(sentiment_features)
            
            # TF-IDF features
            if hasattr(preprocessor.named_transformers_['sentence_tfidf'], 'get_feature_names_out'):
                tfidf_features = preprocessor.named_transformers_['sentence_tfidf'].get_feature_names_out()
                feature_names.extend(tfidf_features[:50])  # Limit to top 50 TF-IDF features
            
            # Environmental Score
            feature_names.append('Environmental Score')
            
            # Get feature importances
            importances = self.model.named_steps['regressor'].feature_importances_
            
            # Get top 15 features
            if len(feature_names) == len(importances):
                sorted_indices = np.argsort(importances)[-15:]
                top_features = {
                    feature_names[i]: float(importances[i]) 
                    for i in sorted_indices
                }
            else:
                # Fallback: use generic feature names
                top_features = {
                    f'Feature_{i}': float(imp) 
                    for i, imp in enumerate(importances[-15:])
                }
            
            return {'success': True, 'feature_importance': top_features}
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global instance
esg_score_model = ESGScoreModel()
