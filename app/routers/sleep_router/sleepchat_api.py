from fastapi import APIRouter
from app.services.sleep_services.mongo_service import get_user_chats
from fastapi import APIRouter
from app.graphs.sleep_graph import sleep_graph_general, sleep_graph_daily, sleep_graph_weekly, SleepState

router = APIRouter(prefix="/sleepchat", tags=["Chat"])

#일상 대화
@router.post("/message")
def chat_general(req: dict):
    state = SleepState(req=req, user_id=req.get("user_id"), messages=[], response="")
    result = sleep_graph_general.invoke(state, start="general_chat")
    return {"response": result["response"]}

#일간 리포트
@router.get("/report/daily/{user_id}")
def daily_report(user_id: int):
    state = SleepState(req={}, user_id=user_id, messages=[], response="")
    result = sleep_graph_daily.invoke(state, start="daily_report")
    return {"report": result["response"]}

#주간 리포트
@router.get("/report/weekly/{user_id}")
def weekly_report(user_id: int):
    state = SleepState(req={}, user_id=user_id, messages=[], response="")
    result = sleep_graph_weekly.invoke(state, start="weekly_report")
    return {"report": result["response"]}

#대화 기록 불러오기
@router.get("/history/{user_id}")
def get_chat_history(user_id: int):
    """특정 유저의 최근 대화 기록"""
    chats = get_user_chats(user_id)
    return {"user_id": user_id, "history": chats}
