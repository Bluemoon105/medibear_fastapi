from sqlalchemy import Column, Integer, Float, Date, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DailyActivity(Base):
    __tablename__ = "daily_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    date = Column(Date)
    sleep_hours = Column(Float)
    caffeine_mg = Column(Float)
    alcohol_consumption = Column(Float)
    physical_activity_hours = Column(Float)
    predicted_sleep_quality = Column(Float)
    predicted_fatigue_score = Column(Float)
    recommended_sleep_range = Column(String)
