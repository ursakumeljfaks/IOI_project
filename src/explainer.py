"""
LIME explanation functionality
"""
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer:
    def __init__(self, model, X_train, feature_names):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained HousePriceModel instance
            X_train: Training data (scaled)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
    
    def explain_instance(self, instance, num_features=10):
        """
        Generate LIME explanation for a single instance
        
        Args:
            instance: Single data point (scaled, as numpy array)
            num_features: Number of top features to show
            
        Returns:
            Dictionary with explanation data
        """
        # Ensure instance is 1D array
        if len(instance.shape) > 1:
            instance = instance.flatten()
        
        # Get explanation
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.model.predict,
            num_features=num_features
        )
        
        # Extract feature contributions
        explanation_list = exp.as_list()
        
        # Parse into structured format
        features = []
        impacts = []
        
        for feature_desc, impact in explanation_list:
            # Extract feature name (before the comparison operator)
            feature_name = feature_desc.split('<=')[0].split('>')[0].strip()
            features.append(feature_name)
            impacts.append(impact)
        
        # Get prediction
        prediction = self.model.model.predict(instance.reshape(1, -1))[0]
        
        return {
            'prediction': prediction,
            'features': features,
            'impacts': impacts,
            'explanation_list': explanation_list
        }
    
    def get_top_features(self, explanation, n=5):
        """Get top N most important features"""
        impacts_df = pd.DataFrame({
            'feature': explanation['features'],
            'impact': explanation['impacts']
        })
        impacts_df['abs_impact'] = impacts_df['impact'].abs()
        return impacts_df.nlargest(n, 'abs_impact')
