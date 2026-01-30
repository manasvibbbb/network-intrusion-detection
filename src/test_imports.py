# test_imports.py
from auth.users import UserManager
from database.db_manager import DatabaseManager
from explainability.explainer import ModelExplainer
import joblib, tensorflow as tf

print("✅ All modules import OK")
print("✅ Models load:", joblib.load('models/random_forest.pkl') is not None)
