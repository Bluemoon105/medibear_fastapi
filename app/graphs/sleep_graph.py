from langgraph.graph import StateGraph, START, END
from langgraph.channels import LastValue
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from typing import Annotated, Any, Dict, List
from app.services.sleep_services.user_service import (
    get_daily_activity,
    get_weekly_activity,
    get_user_info,
)
import google.generativeai as genai
from config.sleep_config import settings

# Gemini 모델 초기화
genai.configure(api_key=settings.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# 상태 정의 (user_id → member_no 로 변경)
class SleepState(BaseModel):
    req: Annotated[Dict[str, Any], LastValue(Dict[str, Any])]
    member_no: Annotated[int, LastValue(int)]  
    messages: Annotated[List[Any], add_messages]
    response: Annotated[str, LastValue(str)]

# 유틸 함수
def _safe_avg(items, key, ndigits=None):
    vals = [x.get(key) for x in items if x.get(key) is not None]
    if not vals:
        return "N/A"
    avg = sum(vals) / len(vals)
    return round(avg, ndigits) if ndigits else avg

#일반 대화
def general_chat_node(state: SleepState):
    req = state.req
    message = req.get("message")
    member_no = state.member_no
    user = get_user_info(member_no) or {}

    if not message:
        return {"response": "입력된 질문이 없습니다."}

    prompt = f"""
    너는 수면 코치지만 친구처럼 따뜻하게 조언하는 챗봇이야.
    아래는 사용자의 질문이야. 너무 형식적이지 않게 대화하듯 답변해줘.
    
    [사용자 번호]: {user.get('name', '비공개')}
    [질문]: {message}
    """

    result = model.generate_content(prompt)
    response = result.text.strip()

    return {"response": response, "messages": [AIMessage(content=response)]}

# 일간 리포트
def daily_report_node(state: SleepState):
    member_no = state.member_no
    user = get_user_info(member_no) or {}
    activity = get_daily_activity(member_no) or {}

    if not activity:
        return {"response": "오늘의 수면 데이터가 없습니다."}

    prompt = f"""
    너는 전문 수면 분석가야. 아래 데이터를 기반으로 오늘의 수면 상태를 간단하고 핵심적으로 요약해줘.

    [사용자]
    이름: {user.get('name', '비공개')} / 나이: {user.get('age', '비공개')}세 / 성별: {user.get('gender', '비공개')}

    [오늘 데이터]
    수면시간: {activity.get('sleep_hours', 'N/A')}시간
    피로도: {activity.get('predicted_fatigue_score', 'N/A')}
    수면 질: {activity.get('predicted_sleep_quality', 'N/A')}
    활동량: {activity.get('physical_activity_hours', 'N/A')}시간
    카페인: {activity.get('caffeine_mg', 'N/A')}mg
    알코올: {activity.get('alcohol_consumption', 'N/A')}

    위 데이터를 요약해서 오늘의 컨디션을 3~4문장으로 요약해줘.
    """

    result = model.generate_content(prompt)
    response = result.text.strip()

    return {"response": response, "messages": [AIMessage(content=response)]}

#주간 리포트
def weekly_report_node(state: SleepState):
    member_no = state.member_no
    user = get_user_info(member_no) or {}
    week = get_weekly_activity(member_no) or []

    if not week:
        return {"response": "최근 7일 데이터가 없습니다."}

    avg_sleep = _safe_avg(week, "sleep_hours", 2)
    avg_fatigue = _safe_avg(week, "predicted_fatigue_score", 1)
    avg_quality = _safe_avg(week, "predicted_sleep_quality", 2)
    avg_activity = _safe_avg(week, "physical_activity_hours", 1)

    prompt = f"""
    너는 수면 분석 전문가야. 아래 7일 평균 데이터를 바탕으로 핵심적인 주간 리포트를 작성해줘.

    [사용자]
    이름: {user.get('name', '비공개')} / 나이: {user.get('age', '비공개')}세 / 성별: {user.get('gender', '비공개')}

    [최근 7일 평균]
    수면시간: {avg_sleep}시간
    피로도: {avg_fatigue}
    수면 질: {avg_quality}
    활동량: {avg_activity}시간

    이 데이터를 기반으로,
    1) 사용자의 전반적인 수면 상태 요약
    2) 피로 관리에 대한 간단한 피드백
    3) 향후 1주일 개선 포인트 1~2개를 제시해줘.
    """

    result = model.generate_content(prompt)
    response = result.text.strip()

    return {"response": response, "messages": [AIMessage(content=response)]}


# ----------------------------------------------------------------
# ✅ 그래프 빌드
# ----------------------------------------------------------------
def build_graph(start_node_name, func):
    graph = StateGraph(SleepState)
    graph.add_node(start_node_name, func)
    graph.add_edge(START, start_node_name)
    graph.add_edge(start_node_name, END)
    return graph.compile()


sleep_graph_general = build_graph("general_chat", general_chat_node)
sleep_graph_daily = build_graph("daily_report", daily_report_node)
sleep_graph_weekly = build_graph("weekly_report", weekly_report_node)
