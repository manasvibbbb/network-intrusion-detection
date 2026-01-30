from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
import pandas as pd
import os

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    username = Column(String)
    model_name = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    features = Column(JSON)
    explanation = Column(JSON)

class DatabaseManager:
    def __init__(self, db_path="database/predictions.db"):
        os.makedirs("database", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_prediction(self, username, model_name, prediction, confidence, features, explanation):
        session = self.Session()
        record = PredictionLog(
            username=username,
            model_name=model_name,
            prediction=prediction,
            confidence=confidence,
            features=features,
            explanation=explanation,
        )
        session.add(record)
        session.commit()
        session.close()

    def get_predictions(self, days=7):
        session = self.Session()
        since = datetime.utcnow() - timedelta(days=days)
        rows = session.query(PredictionLog).filter(PredictionLog.timestamp >= since).all()
        session.close()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "username": r.username,
            "model": r.model_name,
            "prediction": r.prediction,
            "confidence": r.confidence
        } for r in rows])
