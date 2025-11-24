from langgraph.graph import StateGraph, START, END
from langgraph.channels import LastValue
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from typing import Annotated, Any, Dict, List, Optional

from app.services.sleep_services.user_service import (
    get_daily_activity,
    get_weekly_activity,
    get_user_info,
    get_member_no_by_email,
)

from groq import Groq
from config.sleep_config import settings

clinet = Groq(api_key=settings.GROQ_API_KEY)

DEFAULT_MODEL = "openai/gpt-oss-120b"

def call_llm(prompt: str, model_name: str = DEFAULT_MODEL):
    def remove_hanja(text: str) -> str:
        import re
        return re.sub(r"[\u4E00-\u9FFF]", "", text)

    response = clinet.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 수면 전문 AI 코치야. "
                    "모든 설명은 한국어로 쉽고 자연스럽게 존댓말로 작성해."
                    "출력에 사용 가능한 문자는 다음으로 제한한다: 한글, 한글 자모, 숫자(0-9), 공백, 기본적인 문장부호(.,!?-).  그 외 유니코드 문자(아랍어, 러시아어, 중국어 등)이 포함되면 잘못된 출력으로 간주하고 다시 생성하라."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    cleaned = remove_hanja(response.choices[0].message.content)
    return cleaned

class SleepState(BaseModel):
    req: Annotated[Dict[str, Any], LastValue(Dict[str, Any])]
    email: Annotated[str, LastValue(str)]
    messages: Annotated[List[Any], add_messages]
    response: Annotated[Optional[str], LastValue(Optional[str])] = None


def _safe_avg(items, key, ndigits=None):
    vals = [x.get(key) for x in items if x.get(key) is not None]
    if not vals:
        return "N/A"
    avg = sum(vals) / len(vals)
    return round(avg, ndigits) if ndigits else avg


# ------------------------ 일반 대화 ------------------------
def general_chat_node(state: SleepState):
    req = state.req
    message = req.get("message")
    email = state.email

    member_no = get_member_no_by_email(email)
    user = get_user_info(member_no) or {}

    if not message:
        return {"response": "입력된 질문이 없습니다."}

    prompt = f"""
    너는 친근한 수면 코치야.

    [사용자]
    이름: {user.get('name')}

    [질문]
    {message}
    """

    response = call_llm(prompt)

    return {"response": response, "messages": [AIMessage(content=response)]}


# ------------------------ 일간 리포트 ------------------------
def daily_report_node(state: SleepState):
    email = state.email
    member_no = get_member_no_by_email(email)

    user = get_user_info(member_no) or {}
    activity = get_daily_activity(member_no) or {}

    if not activity:
        return {"response": "오늘의 수면 데이터가 없습니다."}

    prompt = f"""
    아래는 오늘의 데이터입니다.

    이름: {user.get('name')}
    나이: {user.get('age')} 세
    성별: {user.get('gender')}

    수면시간: {activity.get('sleep_hours')}
    피로도: {activity.get('predicted_fatigue_score')}
    수면 질: {activity.get('predicted_sleep_quality')}
    활동량: {activity.get('physical_activity_hours')}
    카페인: {activity.get('caffeine_mg')}
    알코올: {activity.get('alcohol_consumption')}

    핵심 일간 리포트를 작성해주세요.
    """
    response = call_llm(prompt)

    return {"response": response, "messages": [AIMessage(content=response)]}


# ------------------------ 주간 리포트 ------------------------
def weekly_report_node(state: SleepState):
    email = state.email
    member_no = get_member_no_by_email(email)

    user = get_user_info(member_no) or {}
    week = get_weekly_activity(member_no) or []

    if not week:
        return {"response": "최근 7일간 데이터가 없습니다."}

    avg_sleep = _safe_avg(week, "sleep_hours", 2)
    avg_fatigue = _safe_avg(week, "predicted_fatigue_score", 1)
    avg_quality = _safe_avg(week, "predicted_sleep_quality", 2)
    avg_activity = _safe_avg(week, "physical_activity_hours", 1)

    prompt = f"""
    아래는 최근 7일 평균 데이터입니다.

    이름: {user.get('name')}
    나이: {user.get('age')}
    성별: {user.get('gender')}

    수면시간: {avg_sleep}
    피로도: {avg_fatigue}
    수면 질: {avg_quality}
    활동량: {avg_activity}

    주간 리포트를 작성해주세요.
    """
    response = call_llm(prompt)

    return {"response": response, "messages": [AIMessage(content=response)]}


# ------------------------ 그래프 빌더 ------------------------
def build_graph(start_node_name, func):
    graph = StateGraph(SleepState)
    graph.add_node(start_node_name, func)
    graph.add_edge(START, start_node_name)
    graph.add_edge(start_node_name, END)
    return graph.compile()


sleep_graph_general = build_graph("general_chat", general_chat_node)
sleep_graph_daily = build_graph("daily_report", daily_report_node)
sleep_graph_weekly = build_graph("weekly_report", weekly_report_node)