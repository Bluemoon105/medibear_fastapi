# -*- coding: utf-8 -*-
"""
stress_graph.py (v2 워크플로우 통합 버전)

- 기존 LangGraph 의존성 제거
- YDATA 버전 stress_graph_v2 로직(위기 감지, 구조화 리포트) 이식
- FastAPI 라우터 인터페이스는 유지:
    - DiagnosisState (class)
    - StressDiagnosisGraph.invoke(state)
    - StressInterviewGraph.invoke(state, history)
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field

from app.services.stress_services.ml_service import StressMLService
from app.services.stress_services.dl_emotion_service import EmotionDLService
from app.services.stress_services.llm_service import StressLLMService

print(">>> stress_graph (v2) 모듈 로드됨")

# 서비스 인스턴스
_ml = StressMLService()
_dl = EmotionDLService()
_llm = StressLLMService()

# ==================================
# 상수 / 규칙
# ==================================

NEGATIVE_SET = {"Anxiety", "Sadness", "Anger", "Fear", "Depression", "Stress"}
ESSENTIAL_KEYS = {"sleep", "diet", "activity"}  # 수면, 식이, 활동

RISK_KEYWORDS = [
    "죽고 싶", "죽고싶", "자살", "살기 싫", "살기싫",
    "끝내고 싶", "끝내고싶", "없어졌으면 좋겠", "존재하고 싶지 않",
    "괴로워서 죽", "극단적인 선택",
]


def detect_crisis(text: str) -> bool:
    """아주 단순한 키워드 기반 위기 신호 감지 (한국어)."""
    if not text:
        return False
    t = text.strip()
    return any(kw in t for kw in RISK_KEYWORDS)


def render_crisis_message() -> str:
    """위기 감지 시 보여줄 고정 안내문 (LLM 호출 없이 사용)."""
    return (
        "지금 정말 많이 힘드신 것 같아요.\n\n"
        "이야기해 줘서 고마워요. 혼자 버티려고만 하지 않으셔도 괜찮아요.\n"
        "AI가 도와줄 수 있는 부분도 있지만, **당장 곁에서 도와줄 수 있는 사람**과 "
        "연결되는 게 무엇보다 중요해요.\n\n"
        "가능하다면 믿을 수 있는 가족, 친구, 선생님, 동료에게 지금 마음을 조금만 "
        "나눠 보실 수 있을까요?\n\n"
        "또 아래와 같은 전문 상담 채널도 있어요 (대한민국 기준):\n"
        "- 📞 자살 예방 상담전화: **1393** (24시간, 무료)\n"
        "- 📞 정신건강 상담전화: **1577-0199**\n"
        "- 📞 청소년 전화: **1388**\n\n"
        "지금 느끼는 마음은 절대 가볍지 않고, 도움을 요청할 자격이 충분히 있어요.\n"
        "당신이 여기까지 버텨온 것만으로도 이미 정말 대단하다는 걸 꼭 기억해 주세요."
    )


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split())


# 🔹 LLM 리포트에서 특정 섹션([3], [4] 등)만 뽑아오는 헬퍼
def _extract_llm_section(coaching_text: str, section_no: int) -> List[str]:
    """
    LLM이 만든 전체 리포트에서
    '[3] ...' 처럼 시작하는 섹션만 잘라서 반환.
    """
    if not coaching_text:
        return []

    lines = coaching_text.splitlines()
    headers: List[tuple[int, int]] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("[") or "]" not in stripped:
            continue
        # "[3] ..." 에서 3만 뽑기
        close_idx = stripped.find("]")
        num_str = stripped[1:close_idx]
        if num_str.isdigit():
            headers.append((idx, int(num_str)))

    if not headers:
        return []

    start_idx = None
    end_idx = len(lines)

    for i, (idx, num) in enumerate(headers):
        if num == section_no and start_idx is None:
            start_idx = idx
            # 다음 헤더 직전까지
            if i + 1 < len(headers):
                end_idx = headers[i + 1][0]
            break

    if start_idx is None:
        return []

    section_lines = lines[start_idx:end_idx]

    # 앞/뒤 공백 줄 정리
    while section_lines and not section_lines[0].strip():
        section_lines = section_lines[1:]
    while section_lines and not section_lines[-1].strip():
        section_lines = section_lines[:-1]

    return section_lines


# ==================================
# State 정의
# ==================================


class DiagnosisState(BaseModel):
    # 입력
    user_query: Optional[str] = ""          # 사용자가 적은 코멘트 / 고민
    sleep_hours: Optional[float] = None
    activity_level: Optional[float] = None  # 1~10 등 스케일
    caffeine_cups: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None            # "male"/"female"/...
    audio_bytes: Optional[bytes] = None     # 업로드 음성(선택)

    # 중간 / 출력 (ML/DL)
    stress_score: Optional[float] = None    # 0~100
    stress_level: Optional[str] = None      # low/medium/high
    emotion_state: Optional[str] = None     # Anxiety / Neutral …

    ml_result: Dict[str, Any] = Field(default_factory=dict)
    dl_result: Dict[str, Any] = Field(default_factory=dict)

    # 인터뷰 상태
    interview_turns: int = 0
    interview_data: List[Dict[str, Any]] = Field(default_factory=list)
    next_question: Optional[str] = None     # 인터뷰용 질문

    # v2에서 추가된 상태
    is_data_complete: bool = False          # 데이터 충족 여부 플래그
    max_turns: int = 5                      # 인터뷰 최대 횟수

    # 위기 감지
    is_crisis: bool = False
    crisis_message: str = ""

    # LLM 최종 결과
    diagnosis_summary: Optional[str] = None  # LLM 코칭 텍스트 전체
    report: Optional[str] = None             # 구조화된 최종 보고서 텍스트


# ==================================
# INITIAL_ANALYSIS (ML + DL)
# ==================================


def _infer_stress_level(score: float) -> str:
    """점수 → low/medium/high 단순 규칙."""
    s = float(score)
    if s < 40:
        return "low"
    if s < 70:
        return "medium"
    return "high"


def node_initial_analysis(state: DiagnosisState) -> DiagnosisState:
    """ML/DL 분석 + 위기 키워드 1차 감지."""

    # ===== ML: StressMLService 사용 =====
    sleep = state.sleep_hours if state.sleep_hours is not None else 6.0
    act = state.activity_level if state.activity_level is not None else 3.0
    caf = state.caffeine_cups if state.caffeine_cups is not None else 1.0
    age = state.age if state.age is not None else 30
    gender = (state.gender or "Other").capitalize()  # Male/Female/Other

    features = {
        "age": age,
        "gender": gender,
        "occupation": "Other",
        "sleep_duration": sleep,
        "quality_of_sleep": 3,
        "physical_activity_level": act,
        "bmi_category": "Normal",
        "heart_rate": 75,
        "daily_steps": 5000,
        "bp_sys": 120,
        "bp_dia": 80,
        "caffeine_cups": caf,  # 원래 모델 피처엔 없지만 참고용
    }

    score = float(_ml.predict_as_score(features))  # 0~100
    level = _infer_stress_level(score)

    if level == "low":
        comment = "전반적으로 스트레스가 비교적 낮은 편이에요."
    elif level == "medium":
        comment = "스트레스가 조금씩 쌓이고 있는 상태로 보입니다."
    else:
        comment = "스트레스가 상당히 높은 편이라, 생활 패턴 점검이 필요해 보여요."

    ml_result: Dict[str, Any] = {
        "stress_score_0_100": score,
        "stress_level": level,
        "stress_comment": comment,
        "top_features": [],
    }

    state.stress_score = score
    state.stress_level = level
    state.ml_result = ml_result

    # ===== DL: EmotionDLService 사용 =====
    if state.audio_bytes:
        try:
            label, prob = _dl.predict_emotion_from_bytes(state.audio_bytes)
        except Exception:
            label, prob = "neutral", 0.0
    else:
        label, prob = "neutral", 0.0

    dl_result: Dict[str, Any] = {
        "primary_emotion": label,
        "confidence": prob,
        "probabilities": {},
        "model_meta": {"note": "emotion_cnn_lstm_all.h5"},
    }

    state.dl_result = dl_result
    state.emotion_state = label

    # ===== 위기 키워드 기반 1차 감지 =====
    crisis_hit = detect_crisis(state.user_query or "")
    state.is_crisis = crisis_hit
    if crisis_hit:
        state.crisis_message = render_crisis_message()

    return state


# ==================================
# 구조화 리포트 생성 (짧고 핵심 + 위로)
# ==================================


def _build_structured_report(state: DiagnosisState, coaching_text: str) -> str:
    """
    최종 보고서를 짧고 핵심 중심으로 구성한 버전.

    - [1] 현재 상태 요약
    - [2] 현재 나타나는 주요 패턴 (인터뷰 요약 2~3줄)
    - [3] AI 코칭 제안  → LLM의 [3] + [6] 섹션 묶어서 사용
    - [4] 오늘의 한마디(위로)
    - [5] 한 줄 요약
    - (위기 감지 시 상단에 위기 안내)
    """
    lines: List[str] = []

    # [0] 위기 안내
    if state.is_crisis:
        lines.append("⚠️ [위기 신호 안내]")
        lines.append(state.crisis_message or "")
        lines.append("\n---\n")

    stress_score = state.stress_score or 0.0
    stress_level = state.stress_level or "unknown"
    emotion = state.emotion_state or "unknown"

    # [1] 현재 상태 요약
    lines.append("[1] 현재 상태 요약")
    lines.append(f"- 스트레스 점수: {stress_score:.1f} / 100 ({stress_level})")
    lines.append(f"- 주요 감정: {emotion}")
    lines.append("")

    # [2] 현재 나타나는 주요 패턴
    lines.append("[2] 현재 나타나는 주요 패턴")

    answered = [
        item for item in (state.interview_data or [])
        if item.get("value") not in (None, "", "None")
    ]

    if answered:
        for item in answered[:3]:
            v = item.get("value")
            if v:
                lines.append(f"- {v}")
    else:
        if state.user_query:
            lines.append(f"- {state.user_query}")
        else:
            lines.append("- 추가 인터뷰 정보 없음")
    lines.append("")

    # [3] AI 코칭 제안  👉 LLM의 [3] + [6] 섹션을 가져와서 보여줌
    lines.append("[3] AI 코칭 제안")

    if coaching_text:
        # LLM 리포트에서 [3], [6] 섹션 뽑기
        section3 = _extract_llm_section(coaching_text, 3)  # AI 코칭 제안
        section6 = _extract_llm_section(coaching_text, 6)  # 라이프스타일 실천 팁

        added_any = False

        # 공통 유틸: 맨 앞의 "[n]" 헤더는 제거
        def _strip_header(section_lines: List[str]) -> List[str]:
            cleaned = [ln for ln in section_lines if ln.strip()]
            if cleaned and cleaned[0].lstrip().startswith("["):
                cleaned = cleaned[1:]
            return cleaned

        # (1) [3] 본문 먼저
        if section3:
            body3 = _strip_header(section3)
            if body3:
                lines.extend(body3)
                lines.append("")
                added_any = True

        # (2) [6] 라이프스타일 팁을 서브블록으로
        if section6:
            body6 = _strip_header(section6)
            if body6:
                lines.append("── 라이프스타일 실천 팁 ──")
                lines.extend(body6)
                lines.append("")
                added_any = True

        # (3) 혹시 [3], [6] 둘 다 못 찾았을 때는 LLM 앞부분 일부라도 폴백으로 사용
        if not added_any:
            raw_lines = [ln for ln in coaching_text.strip().split("\n") if ln.strip()]
            lines.extend(raw_lines[:8])  # 앞에서 6~8줄 정도 보여주기
            lines.append("")

        # 공통 마무리 한 줄
        lines.append(
            "→ 위 제안들(특히 오늘 실천해볼 수 있는 것들) 중에서 "
            "**가장 부담 없는 것 한 가지만** 골라 가볍게 시도해 보는 걸 목표로 해보세요."
        )
    else:
        lines.append("코칭 내용이 생성되지 않았습니다.")

    lines.append("")

    # [4] 오늘의 한마디 (위로용 한 줄)
    lines.append("[4] 오늘의 한마디")
    if stress_score >= 70:
        today = (
            "지금 이 시기를 버티고 있는 것만으로도 이미 정말 대단한 일을 해내고 있어요. "
            "오늘 하루만큼은 스스로를 조금 더 다정하게 대해 주세요."
        )
    elif stress_score >= 40:
        today = (
            "요즘 많이 버티고 있다는 것, 그 자체가 이미 큰 노력이라는 걸 잊지 않으셨으면 해요. "
            "잠깐 숨 고를 틈을 자신에게 허락해 주세요."
        )
    else:
        today = (
            "지금까지 잘 해오신 것만큼, 오늘도 '이 정도면 나 정말 잘하고 있어'라고 "
            "자신에게 한 번 말해 보셨으면 해요."
        )
    lines.append(f"- {today}")
    lines.append("")

    # [5] 한 줄 요약
    lines.append("[5] 한 줄 요약")
    if stress_score >= 70:
        summary = (
            "스트레스가 상당히 높은 편이에요. 가장 부담되는 생활 요소 한 가지부터 "
            "조금씩 조절해 보는 것을 추천드려요."
        )
    elif stress_score >= 40:
        summary = (
            "스트레스가 서서히 쌓이고 있어요. 무리가 되는 부분을 하나 정해서 "
            "완화해보면 도움이 될 수 있어요."
        )
    else:
        summary = (
            "현재 스트레스는 비교적 관리 가능한 수준이지만, 피로가 누적되지 않도록 "
            "수면과 휴식을 꾸준히 챙겨주세요."
        )
    lines.append(f"- {summary}")

    return "\n".join(lines)


def node_prescription_generation(state: DiagnosisState) -> DiagnosisState:
    """
    PRESCRIPTION_GENERATION
    - ML/DL + 인터뷰 데이터 + 위기 여부를 payload 로 LLM에 전달
    - LLM이 생성한 코칭 텍스트를 받아 구조화된 보고서로 조립
    """
    payload = {
        "user_query": state.user_query,
        "ml_stress": state.ml_result,
        "dl_emotion": state.dl_result,
        "interview_data": state.interview_data,
        "is_crisis": state.is_crisis,
    }

    coaching_text: str
    try:
        coaching_text = _llm.generate_coaching_with_payload(payload)
    except Exception:
        try:
            coaching_text = _llm.generate_coaching(
                ml_score=state.stress_score or 0.0,
                emotion=state.emotion_state or "neutral",
                user_note=state.user_query or "",
                ml_top_features=None,
                user_info={},
                context={},
            )
        except Exception:
            score = state.stress_score or 0.0
            coaching_text = (
                f"(LLM 폴백) 현재 추정 스트레스 점수는 약 {score:.1f}점입니다. "
                "3분 복식호흡과 10분 가벼운 산책으로 긴장을 조금씩 풀어보는 것을 추천드립니다."
            )

    state.diagnosis_summary = coaching_text
    state.report = _build_structured_report(state, coaching_text)
    return state


# ==================================
# 인터뷰 그래프
# ==================================


def _fallback_question(turns: int, exclude: Optional[str] = None) -> str:
    candidates = [
        "최근 며칠 동안 특히 부담되거나 힘들었던 순간이 있다면 어떤 때였나요?",
        "요즘 마음이 가장 무거워지는 상황은 어떤 때인가요?",
        "하루를 마쳤을 때 가장 지치는 이유는 무엇이라고 느끼세요?",
        "요즘 생활 리듬 중에서 가장 흐트러졌다고 느끼는 부분이 있을까요?",
    ]

    ex = _norm(exclude)
    for c in candidates:
        if _norm(c) != ex:
            return c

    return candidates[0]


def node_interview(
    state: DiagnosisState,
    history: Optional[List[Dict[str, str]]] = None,
) -> DiagnosisState:
    """다음에 물어볼 인터뷰 질문 1개 생성."""
    turns = (state.interview_turns or 0) + 1
    state.interview_turns = turns

    last_question: Optional[str] = None
    if state.interview_data:
        for item in reversed(state.interview_data):
            q = item.get("question")
            if q:
                last_question = q
                break

    state_dict = state.model_dump()

    try:
        question = _llm.generate_interview_question(
            state=state_dict,
            history=history or [],
        )
    except Exception:
        question = None

    if not question:
        question = _fallback_question(turns, exclude=last_question)
    else:
        if _norm(question) == _norm(last_question or ""):
            question = _fallback_question(turns, exclude=last_question)

    state.next_question = question

    data = list(state.interview_data or [])
    data.append(
        {
            "turn": turns,
            "type": "generic",
            "question": question,
            "value": None,
        }
    )
    state.interview_data = data

    return state


# ==================================
# Graph 래퍼 (router 인터페이스용)
# ==================================


class _StressDiagnosisGraphWrapper:
    def invoke(self, state: DiagnosisState) -> DiagnosisState:
        state = node_initial_analysis(state)
        state = node_prescription_generation(state)
        return state


class _StressInterviewGraphWrapper:
    def invoke(
        self,
        state: DiagnosisState,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> DiagnosisState:
        state = node_initial_analysis(state)
        state = node_interview(state, history=history)
        return state


StressDiagnosisGraph = _StressDiagnosisGraphWrapper()
StressInterviewGraph = _StressInterviewGraphWrapper()


# 로컬 테스트용
if __name__ == "__main__":
    print(">>> Running StressDiagnosisGraph (v2 wrapper)")

    base_state = DiagnosisState(
        user_query="요즘 잠도 잘 못 자고, 머릿속이 복잡해서 계속 불안해요.",
        sleep_hours=5,
        activity_level=2,
        caffeine_cups=3,
        age=25,
        gender="female",
        audio_bytes=None,
    )

    diag_out = StressDiagnosisGraph.invoke(base_state)
    print("\n[DIAGNOSIS RESULT]")
    print("Stress Score:", diag_out.stress_score)
    print("Stress Level:", diag_out.stress_level)
    print("Emotion:", diag_out.emotion_state)
    print("\n[REPORT PREVIEW]")
    print((diag_out.report or "")[:400], "...")