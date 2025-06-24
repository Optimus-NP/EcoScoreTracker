import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import logging

class PackagingModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.is_trained = False
        self.feature_names = ['Material_Type', 'Product_Weight_g', 'Fragility', 'Recyclable', 'Transport_Mode', 'LCA_Emission_kgCO2']
        self.categorical_cols = ['Material_Type', 'Fragility', 'Recyclable', 'Transport_Mode']
        
    def create_sample_data(self):
        """Create sample training data for the packaging model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Define possible values
        materials = ['Glass', 'Plastic', 'Metal', 'Paper', 'Cardboard']
        fragility_levels = ['Low', 'Medium', 'High']
        recyclable_options = ['Yes', 'No']
        transport_modes = ['Land', 'Air', 'Sea']
        packaging_options = ['Bubble Wrap', 'Cardboard Box', 'Plastic Container', 'Paper Bag', 'Metal Can']
        
        data = []
        for i in range(n_samples):
            material = np.random.choice(materials)
            fragility = np.random.choice(fragility_levels)
            recyclable = np.random.choice(recyclable_options)
            transport = np.random.choice(transport_modes)
            
            # Generate correlated weight and emissions
            if material == 'Glass':
                weight = np.random.normal(200, 50)
                base_emission = 2.5
            elif material == 'Metal':
                weight = np.random.normal(300, 80)
                base_emission = 3.2
            elif material == 'Plastic':
                weight = np.random.normal(100, 30)
                base_emission = 1.8
            elif material == 'Paper':
                weight = np.random.normal(50, 20)
                base_emission = 0.8
            else:  # Cardboard
                weight = np.random.normal(75, 25)
                base_emission = 1.2
                
            weight = max(10, weight)  # Ensure positive weight
            
            # Adjust emissions based on transport and fragility
            emission_multiplier = 1.0
            if transport == 'Air':
                emission_multiplier += 0.5
            elif transport == 'Sea':
                emission_multiplier += 0.2
                
            if fragility == 'High':
                emission_multiplier += 0.3
            elif fragility == 'Medium':
                emission_multiplier += 0.1
                
            emission = base_emission * emission_multiplier + np.random.normal(0, 0.2)
            emission = max(0.1, emission)
            
            # Choose packaging based on material and fragility
            if fragility == 'High':
                packaging = np.random.choice(['Bubble Wrap', 'Cardboard Box'], p=[0.7, 0.3])
            elif material in ['Glass', 'Metal']:
                packaging = np.random.choice(['Cardboard Box', 'Plastic Container'], p=[0.6, 0.4])
            elif material == 'Paper':
                packaging = np.random.choice(['Paper Bag', 'Cardboard Box'], p=[0.7, 0.3])
            else:
                packaging = np.random.choice(packaging_options)
            
            data.append({
                'Product_ID': f'P{i+1:04d}',
                'Material_Type': material,
                'Product_Weight_g': round(weight, 1),
                'Fragility': fragility,
                'Recyclable': recyclable,
                'Transport_Mode': transport,
                'LCA_Emission_kgCO2': round(emission, 2),
                'Packaging_Option': packaging
            })
        
        return pd.DataFrame(data)
    
    def train(self, df=None):
        """Train the packaging suggestion model"""
        try:
            if df is None:
                df = self.create_sample_data()
                
            logging.info("Training packaging model...")
            
            # Encode categorical columns
            for col in self.categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            
            # Encode target
            self.target_encoder = LabelEncoder()
            df['Packaging_Option'] = self.target_encoder.fit_transform(df['Packaging_Option'])
            
            # Features and target
            X = df.drop(columns=['Product_ID', 'Packaging_Option'])
            y = df['Packaging_Option']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ])
            
            # Hyperparameter grid
            param_grid = {
                'classifier__n_estimators': [100, 150],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 4],
                'classifier__subsample': [0.8, 1.0]
            }
            
            # Grid Search
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            logging.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            
            return {
                'success': True,
                'accuracy': accuracy,
                'best_params': grid_search.best_params_
            }
            
        except Exception as e:
            logging.error(f"Error training packaging model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, input_data):
        """Make packaging suggestion prediction"""
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
                        # Handle unknown categories
                        try:
                            df_input[col] = self.label_encoders[col].transform([input_data[col]])[0]
                        except ValueError:
                            # If category not seen during training, use the most common category
                            df_input[col] = 0
            
            # Make prediction
            prediction = self.model.predict(df_input)[0]
            predicted_packaging = self.target_encoder.inverse_transform([prediction])[0]
            
            # Get prediction probability
            probabilities = self.model.predict_proba(df_input)[0]
            confidence = float(max(probabilities))
            
            return {
                'success': True,
                'prediction': predicted_packaging,
                'confidence': confidence,
                'all_probabilities': {
                    self.target_encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            logging.error(f"Error making packaging prediction: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self):
        """Get feature importance for the trained model"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Get feature importance from the classifier step of the pipeline
            classifier = self.model.named_steps['classifier']
            importances = classifier.feature_importances_
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, importances)
            }
            
            return {'success': True, 'feature_importance': feature_importance}
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global instance
packaging_model = PackagingModel()
