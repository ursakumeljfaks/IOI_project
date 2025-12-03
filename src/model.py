"""
House price prediction model with LIME explanations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'r2': r2,
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
        return self.model.predict(X_scaled)
    
    def save(self, filepath='models/house_price_model.pkl'):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath='models/house_price_model.pkl'):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"✓ Model loaded from {filepath}")
