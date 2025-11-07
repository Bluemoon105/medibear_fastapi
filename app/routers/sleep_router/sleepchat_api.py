from fastapi import APIRouter
from services.sleep_services.llm_service import generate_general_chat, generate_daily_report, generate_weekly_report
from services.sleep_services.mongo_service import get_user_chats
from models.sleep_models.sleepSchema import ChatRequest

router = APIRouter(prefix="/sleepchat", tags=["Chat"])

# 일반 대화
@router.post("/message")
def chat_general(req: ChatRequest):
    """일상 수면 관련 대화"""
    response = generate_general_chat(req)
    return {"response": response}

#일간 리포트
@router.get("/report/daily/{user_id}")
def daily_report(user_id:int):
    """오늘의 수면 리포트"""
    response = generate_daily_report(user_id)
    return {"report":response}

# 주간 리포트
@router.get("/report/weekly/{user_id}")
def weekly_report(user_id: int):
    """최근 7일 평균 기반 주간 리포트"""
    response = generate_weekly_report(user_id)
    return {"report": response}

#대화 기록 불러오기
@router.get("/history/{user_id}")
def get_chat_history(user_id: int):
    """특정 유저의 최근 대화 기록"""
    chats = get_user_chats(user_id)
    return {"user_id": user_id, "history": chats}
