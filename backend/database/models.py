from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    JSON
)
from sqlalchemy.orm import relationship

from backend.database.connection import Base


# Farmers Table
class Farmer(Base):
    __tablename__ = "farmers"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship(
        "PredictionHistory",
        back_populates="farmer",
        cascade="all, delete-orphan"
    )


# Prediction History Table
class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    farmer_id = Column(
        Integer,
        ForeignKey("farmers.id"),
        nullable=False
    )

    prediction_type = Column(String(50), nullable=False)
    input_data = Column(JSON, nullable=False)
    result = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    farmer = relationship(
        "Farmer",
        back_populates="predictions"
    )