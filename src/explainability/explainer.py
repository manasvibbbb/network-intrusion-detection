# src/explainability/explainer.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

class ModelExplainer:
    def __init__(self, model, model_type='tree', feature_names=None):
        """
        Initialize explainer
        model_type: 'tree' for RF/XGBoost, 'deep' for Neural Network
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(41)]
        self.explainer = None
    
    def create_explainer(self, background_data):
        """Create SHAP explainer with background data"""
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:  # Neural network
            self.explainer = shap.DeepExplainer(self.model, background_data)
    
    def explain_prediction(self, sample):
        """Get SHAP values for a single prediction"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        shap_values = self.explainer.shap_values(sample)
        
        # For binary classification, take positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Attack class
        
        return shap_values
    
    def get_top_features(self, shap_values, n=5):
        """Get top N contributing features"""
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Take first sample if batch
        
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-n:][::-1]
        
        features = []
        for idx in top_indices:
            features.append({
                'feature': self.feature_names[idx],
                'importance': abs_shap[idx],
                'contribution_pct': (abs_shap[idx] / abs_shap.sum() * 100)
            })
        
        return features
    
    def plot_feature_importance(self, shap_values, max_display=10):
        """Create feature importance plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-max_display:]
        
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, abs_shap[top_indices], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in top_indices])
        ax.set_xlabel('|SHAP value|', fontsize=12)
        ax.set_title('Top Feature Contributions', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
