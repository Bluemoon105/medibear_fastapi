from pydantic import BaseModel

class UserInput(BaseModel):
    age: int
    gender: int
    caffeine_mg: float
    sleep_hours: float
    physical_activity_hours: float
    alcohol_consumption: float
class ChatRequest(BaseModel):
    member_no: int  
    message: str

class SleepChatRequest(BaseModel):
    member_no: int  
    sleep_quality: float
    fatigue_score: float
    recommended_range: str
    message: str = "오늘 내 수면 상태 어때?"
