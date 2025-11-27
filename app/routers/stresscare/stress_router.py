from typing import Optional, Dict, Any, List, Literal
#stress_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field

from app.services.stress_services.dl_emotion_service import EmotionDLService
from app.services.stress_services.ml_service import StressMLService
from app.services.stress_services.llm_service import StressLLMService

# 그래프 v2 (DiagnosisState + Graph Wrapper) + 위기 유틸
from app.graphs.stress_graph import (
    DiagnosisState,
    StressDiagnosisGraph,
    StressInterviewGraph,
    detect_crisis,
    render_crisis_message,
)

router = APIRouter(prefix="/stress", tags=["stresscare"])

_dl = EmotionDLService()
_ml = StressMLService()
_llm = StressLLMService()


# ============================================================
# 공용 모델
# ============================================================

class ReportIn(BaseModel):
    sleepHours: Optional[float] = Field(
        None, description="전날 수면 시간(시간 단위)"
    )
    activityLevel: Optional[float] = Field(
        None, description="활동 지수(0~10)"
    )
    caffeineCups: Optional[float] = Field(
        None, description="카페인 섭취(잔/일)"
    )
    primaryEmotion: Optional[str] = Field(
        "unknown", description="주요 감정 라벨 (예: happy, sad, angry...)"
    )
    comment: Optional[str] = Field(
        "", description="자유 서술형 코멘트"
    )


class ReportOut(BaseModel):
    stressScore: float
    primaryEmotion: Optional[str]
    coachingText: str
    meta: Dict[str, Any]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    ml: Dict[str, Any] = Field(default_factory=dict)
    dl: Dict[str, Any] = Field(default_factory=dict)
    coaching: str = ""
    history: List[ChatTurn] = Field(default_factory=list)
    question: str


class ChatOut(BaseModel):
    reply: str


# ============================================================
# Agent (인터뷰 기반 LangGraph 래퍼)
# ============================================================

class AgentState(BaseModel):
    sleepHours: Optional[float] = None
    activityLevel: Optional[float] = None
    caffeineCups: Optional[float] = None
    primaryEmotion: Optional[str] = None
    comment: Optional[str] = None
    interviewTurns: int = 0


class AgentStepRequest(BaseModel):
    state: AgentState = Field(default_factory=AgentState)
    message: str
    # history: [{ role: "assistant" | "user", content: string }]
    history: List[Dict[str, str]] = Field(default_factory=list)


class AgentStepResponse(BaseModel):
    mode: Literal["ask", "interview", "final"]
    reply: str
    state: AgentState
    report: Optional[ReportOut] = None
    isCrisis: bool = False


# ============================================================
@router.get("/health")
def health():
    return {"ok": True, "service": "stresscare", "status": "healthy"}


# ============================================================
# 0) Agent Interview Step
# ============================================================

@router.post("/agent/step", response_model=AgentStepResponse)
def agent_step(body: AgentStepRequest):

    print("[/agent/step] ", body.model_dump())

    # 0) 위기 감지
    if detect_crisis(body.message or ""):
        return AgentStepResponse(
            mode="final",
            reply=render_crisis_message(),
            state=body.state,
            report=None,
            isCrisis=True,
        )

    # 현재까지 완료된 인터뷰 턴 수
    prev_turns = body.state.interviewTurns or 0
    current_turn = prev_turns + 1
    MAX_TURNS = 3

    # 1) 인터뷰 진행 중 (최종 리포트 전)
    if current_turn < MAX_TURNS:
        # 그래프에는 "이전 턴 수"를 넘기고, 증가 자체는 node_interview가 담당
        base_state = DiagnosisState(
            user_query=body.state.comment or "",
            sleep_hours=body.state.sleepHours,
            activity_level=body.state.activityLevel,
            caffeine_cups=body.state.caffeineCups,
            interview_turns=prev_turns,
        )

        try:
            # history를 함께 넘겨서 LLM이 직전 대화를 참고하도록
            inter = StressInterviewGraph.invoke(
                base_state,
                history=body.history,
            )
            next_q = inter.next_question or "지금 상황을 조금 더 자세히 설명해 줄 수 있을까?"
            # 그래프에서 증가된 턴을 그대로 사용
            next_turns = inter.interview_turns or current_turn
        except Exception as e:
            print("[StressInterviewGraph 오류]", e)
            next_q = "지금 상황을 조금 더 자세히 설명해 줄 수 있을까?"
            next_turns = current_turn

        # 첫 질문은 mode="ask", 이후는 "interview"
        mode: Literal["ask", "interview"]
        mode = "ask" if prev_turns == 0 else "interview"

        return AgentStepResponse(
            mode=mode,
            reply=next_q,
            state=AgentState(
                sleepHours=body.state.sleepHours,
                activityLevel=body.state.activityLevel,
                caffeineCups=body.state.caffeineCups,
                primaryEmotion=body.state.primaryEmotion,
                comment=body.state.comment,
                interviewTurns=next_turns,
            ),
            report=None,
            isCrisis=False,
        )

    # 2) 인터뷰 종료 → DiagnosisGraph 실행
    #    (history + 마지막 message를 Q/A로 정리)

    interview_items_raw: List[Dict[str, str]] = []
    last_q: Optional[str] = None

    # 기존 히스토리에서 Q/A 추출
    for h in body.history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "assistant":
            last_q = content
        elif role == "user" and last_q:
            interview_items_raw.append(
                {"question": last_q, "answer": content}
            )
            last_q = None

    # 마지막 assistant 질문 + 현재 message 묶기
    if last_q and body.message:
        interview_items_raw.append(
            {"question": last_q, "answer": body.message}
        )

    # DiagnosisState 인터뷰 포맷에 맞게 변환 (value 필드 사용)
    interview_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(interview_items_raw, start=1):
        interview_items.append(
            {
                "turn": idx,
                "type": "generic",
                "question": item.get("question"),
                "value": item.get("answer"),   # stress_graph 쪽에서 읽는 필드
            }
        )

    diag_state = DiagnosisState(
        user_query=body.state.comment or "",
        sleep_hours=body.state.sleepHours,
        activity_level=body.state.activityLevel,
        caffeine_cups=body.state.caffeineCups,
        interview_turns=len(interview_items),
        interview_data=interview_items,
    )

    try:
        diag = StressDiagnosisGraph.invoke(diag_state)
    except Exception as e:
        print("[StressDiagnosisGraph 오류]", e)
        fallback_report = ReportOut(
            stressScore=0.0,
            primaryEmotion=body.state.primaryEmotion,
            coachingText=(
                "지금까지의 대화를 정리하는 중에 문제가 발생했어요. "
                "그래도 지금 느끼는 감정과 하루 패턴을 간단히 적어보면 도움이 될 수 있어요."
            ),
            meta={"interview": interview_items, "error": str(e)},
        )
        return AgentStepResponse(
            mode="final",
            reply=(
                "지금까지 얘기해 준 내용을 정리하는 중 약간의 오류가 있었지만, "
                "간단한 조언을 먼저 전달할게."
            ),
            state=AgentState(
                sleepHours=body.state.sleepHours,
                activityLevel=body.state.activityLevel,
                caffeineCups=body.state.caffeineCups,
                primaryEmotion=body.state.primaryEmotion,
                comment=body.state.comment,
                interviewTurns=len(interview_items),
            ),
            report=fallback_report,
            isCrisis=False,
        )

    report = ReportOut(
        stressScore=float(diag.stress_score or 0),
        primaryEmotion=diag.emotion_state or body.state.primaryEmotion,
        coachingText=diag.report or diag.diagnosis_summary or "",
        meta={
            "ml": getattr(diag, "ml_result", None),
            "dl": getattr(diag, "dl_result", None),
            "interview": interview_items,
            "is_crisis": diag.is_crisis,
            "crisis_message": diag.crisis_message,
            "source": "StressDiagnosisGraph.v2",
        },
    )

    return AgentStepResponse(
        mode="final",
        reply="지금까지 얘기해 준 내용을 기반으로 리포트를 정리했어! 😊",
        state=AgentState(
            sleepHours=body.state.sleepHours,
            activityLevel=body.state.activityLevel,
            caffeineCups=body.state.caffeineCups,
            primaryEmotion=report.primaryEmotion,
            comment=body.state.comment,
            interviewTurns=len(interview_items),
        ),
        report=report,
        isCrisis=bool(diag.is_crisis),
    )


# ============================================================
# 1) DL 감정 분석
# ============================================================

@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        label, prob = _dl.predict_emotion_from_bytes(raw)
        return {"emotion": label, "confidence": prob}
    except Exception as e:
        print("[/stress/audio 오류]", e)
        raise HTTPException(status_code=400, detail=f"audio error: {e}")


# ============================================================
# 2) 통합 리포트(JSON + ML/DL/LLM) - 그래프 기반
# ============================================================

@router.post("/report/json", response_model=ReportOut)
def make_report_json(body: ReportIn):

    if body.sleepHours is None or body.activityLevel is None or body.caffeineCups is None:
        raise HTTPException(422, "필수 입력값 누락 (sleepHours, activityLevel, caffeineCups)")

    # LangGraph v2 진단 그래프 사용
    diag_state = DiagnosisState(
        user_query=body.comment or "",
        sleep_hours=body.sleepHours,
        activity_level=body.activityLevel,
        caffeine_cups=body.caffeineCups,
        # age, gender 필요하면 나중에 추가
    )

    try:
        diag = StressDiagnosisGraph.invoke(diag_state)

        report = ReportOut(
            stressScore=float(diag.stress_score or 0.0),
            primaryEmotion=diag.emotion_state or body.primaryEmotion,
            coachingText=diag.report or diag.diagnosis_summary or "",
            meta={
                "ml": getattr(diag, "ml_result", None),
                "dl": getattr(diag, "dl_result", None),
                "interview": getattr(diag, "interview_data", None),
                "is_crisis": diag.is_crisis,
                "crisis_message": diag.crisis_message,
                "source": "StressDiagnosisGraph.v2",
            },
        )
        return report

    except Exception as e:
        # LangGraph 전체 실패 시 예전 ML + LLM 방식으로 폴백
        print("[/report/json StressDiagnosisGraph 오류, fallback 사용]", e)

        try:
            stress_score = float(
                _ml.predict_as_score(
                    {
                        "sleep_duration": body.sleepHours,
                        "physical_activity_level": body.activityLevel,
                        "caffeine_cups": body.caffeineCups,
                    }
                )
            )
        except Exception as e2:
            print("[/report/json ML fallback 오류]", e2)
            stress_score = 0.0

        try:
            coaching = _llm.generate_coaching(
                ml_score=stress_score,
                emotion=body.primaryEmotion,
                user_note=(body.comment or ""),
            )
        except Exception as e3:
            print("[/report/json LLM fallback 오류]", e3)
            coaching = (
                f"(fallback) 현재 추정 스트레스 점수는 {stress_score:.1f}입니다. "
                f"오늘은 3분 정도 깊은 복식호흡과 가벼운 스트레칭으로 몸을 풀어보는 걸 추천해요."
            )

        return ReportOut(
            stressScore=stress_score,
            primaryEmotion=body.primaryEmotion,
            coachingText=coaching,
            meta={"note": body.comment, "error": str(e), "source": "fallback-ml-llm"},
        )


# 2-1) Spring(FormData)용 /report/agent
@router.post("/report/agent", response_model=ReportOut)
def make_report_agent(
    sleepHours: float = Form(...),
    activityLevel: float = Form(...),
    caffeineCups: float = Form(...),
    primaryEmotion: str = Form("unknown"),
    comment: str = Form(""),
):
    """
    Spring에서 multipart/form-data로 호출하는 /stress/report/agent 를
    내부적으로 JSON 버전(/report/json)의 로직에 연결하는 어댑터.
    """
    body = ReportIn(
        sleepHours=sleepHours,
        activityLevel=activityLevel,
        caffeineCups=caffeineCups,
        primaryEmotion=primaryEmotion,
        comment=comment,
    )
    return make_report_json(body)


# ============================================================
# 3) 자유 LLM 챗봇
# ============================================================

@router.post("/chat", response_model=ChatOut)
def free_chat(body: ChatIn):
    # 간단 위기 감지 (질문 텍스트 기준)
    if detect_crisis(body.question):
        return ChatOut(reply=render_crisis_message())

    # 컨텍스트(ML/DL/이전 코칭)를 한 번에 요약해서 넘김
    ctx = {
        "ml": body.ml,
        "dl": body.dl,
        "coaching": body.coaching,
    }

    messages: List[Dict[str, str]] = []

    # 현재 상태 요약 프롬프트
    messages.append(
        {
            "role": "user",
            "content": (
                "다음 JSON은 지금 내 상태 요약이야. 이걸 참고해서 너무 무겁지 않은 톤으로 한국어로만 대화해줘.\n"
                f"{ctx}"
            ),
        }
    )

    # 기존 대화 히스토리
    for t in body.history:
        messages.append({"role": t.role, "content": t.content})

    # 이번 질문
    messages.append({"role": "user", "content": body.question})

    try:
        reply = _llm.chat(messages=messages)
    except Exception as e:
        print("[/stress/chat LLM 오류]", e)
        reply = "지금 잠깐 대화 엔진에 문제가 생긴 것 같아. 잠시 후에 다시 시도해줄래?"

    return ChatOut(reply=reply)