# -*- coding: utf-8 -*-
"""
llm_service.py (StressCare LLM 서비스 - Groq/OpenAI 호환)
- 스트레스 리포트(코칭) + 코칭 챗봇 + 인터뷰 질문(ask 모드) 지원
- 환경변수:
  LLM_API_KEY, LLM_BASE_URL(기본 https://api.groq.com/openai/v1), LLM_MODEL(기본 llama-3.1-8b-instant)
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _clean_repetitive_sections(text: str) -> str:
    """
    반복 문장 / 단락 줄이기:
    - 같은 줄 반복 제거
    - '예를 들어/예를 들면' 연속 중복 제거
    - 앞부분(20글자) 동일 문장 연속 시 첫 문장만 유지
    """
    if not text:
        return text

    # 줄 단위 중복 제거
    lines = text.splitlines()
    seen = set()
    cleaned: List[str] = []
    prev_prefix = None

    for line in lines:
        norm = re.sub(r"\s+", " ", line).strip()

        # 완전 빈 줄은 그대로 두되, 중복 체크는 하지 않음
        if norm == "":
            cleaned.append(line)
            prev_prefix = None
            continue

        # '예를 들어' 계열 반복 줄이기
        if norm.startswith(("예를 들어", "예를 들면")):
            if cleaned:
                last_norm = re.sub(r"\s+", " ", cleaned[-1]).strip()
                if last_norm.startswith(("예를 들어", "예를 들면")):
                    continue

        # 완전 동일 문장은 한 번만
        if norm in seen:
            continue
        seen.add(norm)

        # 앞 20글자 기준으로 비슷한 문장 반복 방지
        prefix = norm[:20]
        if prev_prefix is not None and prefix == prev_prefix:
            continue

        cleaned.append(line)
        prev_prefix = prefix

    text = "\n".join(cleaned).strip()

    # 너무 많은 연속 빈 줄 정리 (최대 2줄)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _extract_lifestyle_tips(text: str) -> str:
    """
    LLM이 반환한 전체 보고서 텍스트에서
    [6] 라이프스타일 실천 팁 영역만 뽑아서,
    - 로 시작하는 bullet 들만 모아 반환한다.

    실패하면 원본 text를 그대로 반환한다.
    """
    if not text:
        return text

    # [6] 섹션 시작 위치 찾기
    # 예: "[6] 라이프스타일 실천 팁" 이라고 나올 것
    pattern = r"\[6\].*라이프스타일\s*실천\s*팁"
    m = re.search(pattern, text)
    if not m:
        # 혹시 영문/다른 형식일 때를 대비해 단순히 "[6]"만 찾는 fallback
        idx = text.find("[6]")
        if idx == -1:
            return text
        block = text[idx:]
    else:
        block = text[m.start():]

    # [7] 같은 다음 섹션이 없으니, block 전체에서 bullet만 추출
    lines = [ln.strip() for ln in block.splitlines()]
    tips: List[str] = []

    for ln in lines:
        # "- "로 시작하는 라인만 실천 팁으로 간주
        if ln.startswith("-"):
            tips.append(ln)

    # bullet이 2개 이상이면 이걸 쓰고, 아니면 원본 유지
    if len(tips) >= 2:
        return "\n".join(tips)

    return text


# ===== 프롬프트들 =====

# 1) 리포트(코칭)용 시스템 프롬프트
_SYSTEM_PROMPT = """
너는 개인 사용자의 스트레스와 감정을 함께 살펴보고,
부드럽고 따뜻한 말투로 현실적인 조언을 해주는 "헬스케어 감정 & 스트레스 코치" AI이다.

[역할]
- 사용자의 생활 패턴과 감정 상태를 이해하고,
  현재 상태를 짧게 정리한 뒤, 실천 가능한 작은 변화를 제안한다.
- 사용자를 평가하거나 훈계하지 않고, 옆에서 함께 고민해 주는 친한 친구 같은 느낌으로 이야기한다.
- 특히 사용자의 감정(슬픔, 분노, 불안, 무기력 등)을 존중하고,
  그 감정이 왜 자연스러운지 먼저 인정해 준 뒤에 코칭을 제안한다.

[입력]
JSON 형식 데이터:
- ml_stress: 머신러닝 스트레스 예측 결과 (0~100 점수, 수준, 코멘트 등)
- dl_emotion: 딥러닝 음성 감정 분석 결과 (주요 감정, 확신도 등)
- user, context: 사용자의 기본 정보와 상황 설명

[반드시 지켜야 할 규칙]
1) 한국어로만 작성한다.
2) 전문 용어보다 쉬운 일상 언어를 사용한다.
3) 부정 감정이나 고스트레스(높음)인 경우,
   - 첫 문단에서 사용자의 감정을 충분히 공감해 주고
   - 그 다음에 구체적인 실천 팁을 제시한다.
4) 의료 진단, 약품명, 처방, 병명 추측은 절대 하지 않는다.
5) 운동·식이 조언은 항상 "무리하지 않는 선에서", "몸 상태를 보면서"와 같이 안전 문장을 함께 넣는다.
6) 과장된 위로나 상투적인 문구(“완벽해요”, “걱정 마세요” 등)는 피하고,
   차분하지만 따뜻한 톤을 유지한다.
7) 동일하거나 매우 유사한 문장을 반복해서 쓰지 않는다.
8) 자살, 자해, 극단적 표현이 있을 경우:
   - 구체적인 방법이나 계획을 절대 언급하지 않는다.
   - 혼자가 아니라는 점과, 주변의 도움(가족/친구/전문가/상담기관)을 구하도록 부드럽게 권한다.
   - 한국(대한민국)의 전문 상담 채널(예: 자살 예방 상담전화 1393 등)을 간단히 안내해도 좋다.

[출력 형식]  ※ 이 섹션 제목과 구조를 그대로 지켜라.

[1] 현재 상태 요약
- 1~3문장으로 사용자의 현재 상황과 전반적인 상태를,
  감정을 포함해서 따뜻하게 정리한다.
  (예: "요즘 정말 많이 버티고 계신 것 같아요. ○○ 때문에 마음이 무거운 날이 계속 이어지고 있어요." 같은 느낌)

[2] 현재 나타나는 주요 패턴
- 사용자의 말, 생활 패턴, 감정 흐름에서 눈에 띄는 부분 2~4가지를 bullet로 정리한다.
- 단순 나열이 아니라, "요즘 ○○한 날이 많다", "특히 △△ 상황에서 힘들어 보인다"처럼 문장 형태로 쓴다.

[3] AI 코칭 제안
- 첫 1~2문장은 사용자의 입장에서 공감과 지지를 표현한다.
- 그 다음 2~4문장은 너무 무겁지 않은 톤으로, 현실적인 방향성을 제안한다.
- "이렇게 하셔야 합니다" 보다는 "이런 방식도 한 번 해보면 어떨까요?"처럼 제안형으로 말한다.

[4] 오늘의 한마디
"사자성어 또는 짧은 명언/속담 한 줄"
- 사용자의 상황에 맞게 선택하고, 너무 뻔한 문구는 피한다.
- 가능한 한 감정에 딱 맞는 문장을 고른다.
한 줄 해석 (쉽게 풀이)
- 위 문장을 사용자의 현재 상황과 연결해서, 공감이 느껴지도록 설명한다.

[5] 한 줄 요약
- 오늘 대화에서 가장 핵심이 되는 메시지를 한 문장으로 정리한다.
- 사용자가 스스로를 조금 덜 탓하고, 조금 더 다정하게 대할 수 있도록 돕는 문장을 쓴다.

[6] 라이프스타일 실천 팁
아래 네 가지 카테고리에서 각각 1개씩, 총 4개의 실천 팁을 bullet 형식으로 제시한다.

수면
- 오늘 또는 이번 주에 바로 해볼 수 있는, 현실적인 수면 관련 제안 1개 (예: 잠들기 전 10분 루틴, 휴대폰 사용 줄이기 등)

운동/신체 활동
- 거창한 운동 계획이 아니라, "10분 산책", "가벼운 스트레칭"처럼 부담이 적은 활동 1개를 제안한다.
- 항상 "몸 상태를 보면서 무리하지 않는 선에서"라는 뉘앙스를 포함한다.

식습관/카페인
- 카페인, 야식, 물 섭취 등과 관련해 오늘부터 조절해 볼 수 있는 작은 시도 1개를 제안한다.
- "완벽하게 끊기"가 아니라 "조금 줄여보기", "시간대를 조정해보기" 같은 현실적인 방향으로 쓴다.

관계/생활 관리
- 주변 사람과의 소통, 스스로에 대한 태도, 하루 마무리 습관 등에서 해볼 수 있는 행동 1개를 제안한다.
- "도움을 요청해도 괜찮다", "오늘 스스로를 한 번 칭찬해 보기"처럼 정서적 지지를 담는다.

[중요]
- 위에서 정의한 [1]~[6] 섹션 제목을 그대로 사용하여, 모두 순서대로 출력해라.
- 특히 [6] 라이프스타일 실천 팁에서는 '수면', '운동/신체 활동', '식습관/카페인', '관계/생활 관리' 네 소제목을 모두 포함해라.
"""

_USER_PROMPT_TMPL = """다음은 한 사용자의 스트레스 예측 결과와 음성 감정 분석 결과이다.
이 정보를 바탕으로 위 System 안내에 나온 형식 그대로 코칭 멘트를 작성해줘.

```json
{json_block}
```"""

# 2) 코칭 챗봇(자유 대화)용 시스템 프롬프트
_CHAT_SYSTEM_PROMPT = """
너는 사용자의 스트레스와 감정을 이미 어느 정도 파악한 상태에서,
친구 같지만 예의를 갖춘 톤으로 대화를 이어가는 "헬스케어 감정 & 스트레스 코치"이다.

[역할]
- 사용자의 고민과 감정을 먼저 들어주고, 짧게 공감한 뒤,
  너무 무겁지 않은 톤으로 작은 제안을 해주는 것이 목적이다.
- 진단 결과(스트레스 점수, 감정 상태, 코칭 요약 등)는 참고만 하고,
  사용자의 '현재 발언'을 가장 중요하게 여긴다.

[답변 스타일]
1) 한국어 존댓말 사용. 하지만 말투는 상담사처럼 딱딱하지 않고,
   친한 친구가 조심스럽게 이야기해 주는 느낌으로 쓴다.
   - 예: "~하셨겠어요"보다는 "~하셨을 것 같아요", "~이지 않을까요?" 같은 부드러운 표현
   - 가끔 "솔직히", "사실은", "조금만" 같은 일상 표현 사용 가능
2) 3~7문장 정도의 길이:
   - 1~2문장: 사용자의 말에 대한 공감 / 정리
   - 2~4문장: 상황에 맞는 구체적인 제안 또는 생각 정리
   - 마지막 1문장: 자연스러운 가벼운 질문 1개로 대화를 이어가기
3) 보고서/항목/번호 매기기 형식은 피하고, 대화하듯이 쓴다.
4) 필요하면 짧은 명언이나 사자성어를 한 줄 정도 가볍게 인용해도 되지만, 필수는 아니다.
5) 같은 문장을 반복해서 쓰지 말고, 이미 한 말을 그대로 다시 설명하지 않는다.

[안전 규칙]
- 사용자가 "죽고 싶다", "자해", "극단적인 선택" 등 심각한 표현을 사용하면:
  - 구체적인 방법, 계획, 수단에 대해 절대 언급하지 않는다.
  - 혼자가 아니라는 점, 도움을 요청해도 된다는 점을 강조한다.
  - 가까운 사람(가족/친구)이나 전문 상담사, 정신건강의학과, 지역 상담 기관 등
    실제 도움을 줄 수 있는 사람/기관과 연결되도록 부드럽게 권한다.
- 법률, 의학, 재정 같은 전문 영역은 조심스럽게 말하고,
  반드시 전문가와 상의하라는 뉘앙스를 포함한다.

[출력 형식]
- 문단 2~4개 정도의 자연스러운 대화체 한국어.
- 마지막 문장은 사용자가 편하게 이어서 이야기할 수 있는 질문으로 끝나도록 한다.
"""

# 3) 인터뷰(ask 모드) 전용 시스템 프롬프트
_INTERVIEW_SYSTEM_PROMPT = """
너는 사용자의 스트레스·감정 상태를 자연스럽게 탐색하는
한국어 심리 인터뷰 코치이다.

[역할]
- 사용자의 마음 상태, 생활 패턴, 스트레스 원인을 파악하기 위해
  ‘대화를 이어가는 느낌’으로 한 번에 질문 하나만 한다.
- 질문은 이전 대화(history)를 참고하여, 이미 한 질문을 반복하지 않도록 한다.

[질문 규칙]
1) 질문은 딱 1문장, 존댓말, 물음표로 끝난다.
   - 말투는 상담사처럼 딱딱하지 않고, 친한 친구가 조심스럽게 물어보는 느낌으로 한다.
2) 조언, 분석, 요약, 평가를 하지 않는다. 오직 질문만 한다.
3) 아래 유형 중 상황에 맞는 질문 한 가지만 선택한다.
   - 감정 탐색: "그때 느끼신 감정은 어떤 느낌에 가까웠나요?"
   - 원인 탐색: "그 일이 발생한 배경을 조금 더 설명해주실 수 있을까요?"
   - 변화 탐색: "요즘 상황이 이전과 달라진 부분이 있나요?"
   - 욕구 탐색: "지금 가장 바라고 계신 점은 무엇인가요?"
   - 대처 탐색: "요즘 힘들 때 어떤 방식으로 풀려고 해보셨나요?"

4) 다음 상황에서는 우선순위를 가진다.
   - 사용자가 부정 감정(힘들다/짜증/우울/불안 등)을 말함 → 감정 탐색 질문
   - 스트레스의 원인을 언급함 → 원인 탐색 질문
   - "잘 모르겠다"라고 답함 → 욕구 탐색 또는 변화 탐색 질문
   - 단답형(한 단어/아주 짧은 답) → 상황을 더 구체적으로 묻는 질문

5) 절대 하면 안 되는 것
   - 같은 질문 또는 거의 같은 구조의 질문을 반복하는 것
   - 조언, 해석, 위로나 분석을 섞는 것
   - 사용자의 답을 부정하거나 평가하는 것

[출력 형식]
- 질문 1문장만 출력 (앞뒤 불필요한 설명 없이)
"""


def _build_payload(
    user_info: Dict[str, Any],
    context: Dict[str, Any],
    ml_stress: Dict[str, Any],
    dl_emotion: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM 입력용 JSON payload 생성"""
    return {
        "user": user_info,
        "context": context,
        "ml_stress": ml_stress,
        "dl_emotion": dl_emotion,
        "app": {"name": "StressCare AI", "locale": "ko-KR", "version": "0.1.0"},
    }


class StressLLMService:
    """
    - 보고서(코칭) 생성: generate_coaching(...)
    - 채팅 응답 생성: chat(...)
    - 인터뷰 질문 생성: generate_interview_question(...)
    """

    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("❌ .env에 LLM_API_KEY가 없습니다. (.env 확인)")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # --------- 인터뷰 질문(ask 모드) ----------

    def generate_interview_question(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, str]] = None,
        temperature: float = 0.4,
        max_tokens: int = 120,
    ) -> str:

        summary_lines: List[str] = []

        sh = state.get("sleep_hours") or state.get("sleepHours")
        al = state.get("activity_level") or state.get("activityLevel")
        cc = state.get("caffeine_cups") or state.get("caffeineCups")
        emo = state.get("emotion_state") or state.get("primaryEmotion")

        if sh is not None:
            summary_lines.append(f"- 수면 시간: {sh}시간")
        if al is not None:
            summary_lines.append(f"- 활동량: {al} (1~10)")
        if cc is not None:
            summary_lines.append(f"- 카페인: {cc}잔/일")
        if emo:
            summary_lines.append(f"- 감정 상태: {emo}")

        state_summary = "\n".join(summary_lines) or "- 상태 정보가 거의 없습니다."

        # 직전 user 발화 탐색
        last_user_message = ""
        if history:
            for h in reversed(history):
                if h.get("role") == "user":
                    last_user_message = h.get("content", "")
                    break

        user_content = (
            "다음은 지금까지 파악된 사용자 상태 정보입니다:\n"
            f"{state_summary}\n\n"
        )
        if last_user_message:
            user_content += f"사용자가 방금 이렇게 말했습니다:\n\"{last_user_message}\"\n\n"

        user_content += (
            "이 정보를 바탕으로, 지금 사용자에게 물어보면 좋을 "
            "'다음 한 가지 질문'만 존댓말 한 문장으로 만들어 주세요."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _INTERVIEW_SYSTEM_PROMPT}
        ]

        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        question = (resp.choices[0].message.content or "").strip()

        # ? 강제 보정
        if not question.endswith(("?", "？")):
            question = question.rstrip("。.!") + "?"

        if len(question) > 120:
            question = question[:110].rstrip() + "?"

        # 중복 질문 방지 로직 추가
        prev_questions = []
        if history:
            for h in history:
                if h.get("role") == "assistant":
                    c = (h.get("content") or "").strip()
                    if c.endswith(("?", "？")):
                        prev_questions.append(c)

        # 공백 normalize
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        normalized_prev = {norm(q) for q in prev_questions}
        normalized_new = norm(question)

        # 이미 같은 질문이 존재하면 fallback 질문로 교체
        if normalized_new in normalized_prev:
            fallback_candidates = [
                "최근 며칠 동안 특히 부담되거나 힘들었던 순간이 있다면 어떤 때였나요?",
                "요즘 마음이 무거워지는 순간이 있다면 언제인가요?",
                "가장 많이 신경 쓰이는 일은 어떤 부분인가요?",
                "요즘 생활 리듬 중에서 가장 흐트러졌다고 느끼는 부분이 있을까요?",
            ]
            for fb in fallback_candidates:
                if norm(fb) not in normalized_prev:
                    question = fb
                    break

        return question

    # --------- 간이 코칭(리포트) ----------

    def generate_coaching(
        self,
        ml_score: float,
        emotion: str,
        user_note: str = "",
        *,
        ml_top_features: Optional[List[Dict[str, Any]]] = None,
        user_info: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        debug: bool = False,
    ) -> str:
        """
        /stress/report 에서 바로 호출하기 좋은 간이 API.
        - ml_score(0~100), emotion(str), user_note(str)만으로 payload 자동 구성
        - 필요하면 ml_top_features / user_info / context 도 추가 가능
        """
        # 범주화(낮음/보통/높음)
        level_ko, level_en = self._level_from_score(ml_score)

        ml = {
            "stress_score_0_100": float(ml_score),
            "stress_score_raw": round(float(ml_score) / 10.0, 2),  # 대략 0~10 스케일
            "stress_level": level_en,
            "stress_level_ko": level_ko,
            "stress_comment": self._comment_from_level(level_ko),
            "top_features": ml_top_features or [],
            "model_meta": {"model_type": "random_forest_bundle"},
        }
        dl = {
            "primary_emotion": str(emotion),
            "confidence": None,
            "probabilities": None,
            "model_meta": {"model_type": "cnn_bilstm_hybrid"},
        }
        ui = user_info or {"notes": user_note}
        ctx = context or {}

        payload = _build_payload(ui, ctx, ml, dl)
        return self._generate_coaching_text(payload, temperature, max_tokens, debug)

    # --------- 풀 payload 코칭 ----------

    def generate_coaching_with_payload(
        self,
        payload: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 1200,
        debug: bool = False,
    ) -> str:
        return self._generate_coaching_text(payload, temperature, max_tokens, debug)

    # --------- 코칭 챗봇(자유 대화) ----------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str:
        """
        코칭 챗봇용 대화 API.

        messages 예:
        [
          {"role":"user","content":"다음 JSON은 컨텍스트야 ... {..}"},
          {"role":"assistant","content":"..."},
          {"role":"user","content":"요즘 잠이 너무 안 와요."}
        ]

        - /stress/chat 라우터에서:
          - 컨텍스트 JSON을 하나의 user 메시지로 보내고
          - 이어서 기존 history + 최신 질문을 전달하면 됨.
        """
        full_messages: List[Dict[str, str]] = [
            {"role": "system", "content": _CHAT_SYSTEM_PROMPT}
        ]
        full_messages.extend(messages)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # 후처리: 반복 문장 정리
        cleaned = _clean_repetitive_sections(raw)

        # 혹시 영어 안내 문구, 역할 태그 등이 섞이면 제거를 유도
        # (프롬프트에서 충분히 막고 있으므로 최소한만 처리)
        return cleaned

    # ===== 내부 구현 =====

    def _generate_coaching_text(
        self,
        payload: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        debug: bool,
    ) -> str:
        if debug:
            print("[DEBUG] Payload]", json.dumps(payload, ensure_ascii=False, indent=2))

        user_prompt = _USER_PROMPT_TMPL.format(
            json_block=json.dumps(payload, ensure_ascii=False, indent=2)
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # 1차: 반복/노이즈 정리
        cleaned = _clean_repetitive_sections(raw)
        # 2차: [6] 라이프스타일 실천 팁만 뽑아서, stress_graph 쪽에서
        #      "이 중에서 가장 부담 없는 것 한 가지만..." 과 딱 맞게 사용
        tips_only = _extract_lifestyle_tips(cleaned)
        return tips_only

    @staticmethod
    def _level_from_score(score: float) -> (str, str):
        """0~100 스코어 → ('낮음'|'보통'|'높음', 'low'|'medium'|'high')"""
        s = float(score)
        if s <= 33:
            return "낮음", "low"
        if s <= 66:
            return "보통", "medium"
        return "높음", "high"

    @staticmethod
    def _comment_from_level(level_ko: str) -> str:
        if level_ko == "낮음":
            return (
                "일상적인 수준의 스트레스예요. 현재 페이스를 너무 바꾸기보다는, "
                "지금의 좋은 습관을 가볍게 유지해 보시면 좋겠어요."
            )
        if level_ko == "보통":
            return (
                "스트레스가 조금 쌓여 있는 상태예요. 잠, 식사, 움직임 중에서 하나만 골라서 "
                "오늘 조금 더 신경 써보면 도움이 될 수 있어요."
            )
        # 높음
        return (
            "스트레스가 꽤 높은 편으로 보이네요. 혼자 너무 버티려고 하기보다는, "
            "가까운 사람이나 전문가와 상의하면서 작은 것부터 함께 조정해 보면 좋겠어요."
        )
