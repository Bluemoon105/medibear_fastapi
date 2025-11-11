from pydantic import BaseModel

class UserInput(BaseModel):
    age: int
    gender: int
    caffeine_mg: float
    sleep_hours: float
    physical_activity_hours: float
    alcohol_consumption: float

# 일상 대화
class ChatRequest(BaseModel):
    user_id: str
    message: str

# 수면 데이터 기반 대화
class SleepChatRequest(BaseModel):
    user_id: str
    sleep_quality: float
    fatigue_score: float
    recommended_range: str
    message: str = "오늘 내 수면 상태 어때?"