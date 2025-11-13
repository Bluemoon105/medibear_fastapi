from fastapi import APIRouter
from app.graphs.sleep_graph import (
    sleep_graph_general,
    sleep_graph_daily,
    sleep_graph_weekly,
    SleepState,
)

router = APIRouter(prefix="/sleepchat", tags=["Chat"])

# 일반 대화
@router.post("/message")
def chat_general(req: dict):
    """
    일반 수면 관련 대화 엔드포인트
    """
    member_no = req.get("member_no") 
    state = SleepState(req=req, member_no=member_no, messages=[], response="")
    result = sleep_graph_general.invoke(state, start="general_chat")
    return {"response": result["response"]}


# 일간 리포트
@router.get("/report/daily/{member_no}")
def daily_report(member_no: int):
    """
    특정 회원의 일간 리포트 요청
    """
    state = SleepState(req={}, member_no=member_no, messages=[], response="")
    result = sleep_graph_daily.invoke(state, start="daily_report")
    return {"report": result["response"]}


# 주간 리포트
@router.get("/report/weekly/{member_no}")
def weekly_report(member_no: int):
    """
    특정 회원의 주간 리포트 요청
    """
    state = SleepState(req={}, member_no=member_no, messages=[], response="")
    result = sleep_graph_weekly.invoke(state, start="weekly_report")
    return {"report": result["response"]}
