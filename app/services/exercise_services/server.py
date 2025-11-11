# server.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import numpy as np
import asyncio
import os

# ===== Embedding =====
from sentence_transformers import SentenceTransformer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"   # ✅ 고정 (384차원)
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

def embed(text: str) -> List[float]:
    vec = embed_model.encode(text, normalize_embeddings=True)
    return vec.tolist()

VECTOR_DIM = len(embed_model.encode("dim"))  # ✅ 384

# ===== LLM (Qwen 1.5B GGUF with llama.cpp) =====
from llama_cpp import Llama
MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=max(1, (os.cpu_count() or 2) - 1),
    n_batch=128,
    logits_all=False,
    verbose=False,
    chat_format="chatml",   # <<< 🔧 변경: qwen2 로 정확한 템플릿 사용
)

# ===== FastAPI =====
app = FastAPI(title="MediBear LLM Server (Local Mongo + RAG)")

# ===== MongoDB (로컬만) =====
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
db = client["ai_coach"]
chat_col = db["chat_history"]
profile_col = db["profile"]

# ===== Schemas =====
class ChatInput(BaseModel):
    user_id: str
    message: str

class ChatWithAnalysisInput(BaseModel):
    user_id: str
    message: str
    analysis: Dict[str, Any]

# ===== Utils =====
def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def safe_get_vec(doc) -> Optional[List[float]]:
    """
    DB 하위호환: 'embedding' 또는 'vector' 키 사용.
    차원이 다르면 None 반환.
    """
    vec = doc.get("embedding") or doc.get("vector")
    if not isinstance(vec, list):
        return None
    if len(vec) != VECTOR_DIM:
        return None
    return vec

# ===== 분석값 → 자연어 상태(숫자 직접 노출 금지) =====
def describe_joint(a: Optional[float]) -> str:
    if a is None: return "보통"
    if a > 170:   return "충분히 펴짐"
    if a > 150:   return "대체로 양호"
    if a > 120:   return "조금 더 펴기"
    return "힘 전달 부족"

def describe_back(a: Optional[float]) -> str:
    if a is None: return "보통"
    if a > 40:    return "허리 과신전 경향"
    if a < 15:    return "허리 말림 경향"
    return "중립 유지 양호"

# # ===== System Prompt (되묻기 금지 + 형식 강제) =====
# SYSTEM_PROMPT = (
#     "너는 한국어 퍼스널 트레이너다. 분석 데이터가 이미 제공되며, 절대 되묻지 않는다.\n"
#     "팔꿈치 각도, 허리 각도, 수치, 숫자(°, %, cm 등) 언급 금지. "
#     "숫자를 유추해서 만들어내는 것도 금지. "
#     "항상 감각 중심 표현으로 바꿔 말한다.\n"
#     "문체는 짧고 단호하지만 따뜻하게. 반복/에코 금지.\n\n"
#     "출력 형식은:\n"
#     "① 자세 느낌 요약 (2문장)\n"
#     "② 잘한 점 (1문장)\n"
#     "③ 개선할 점 (2~3개 불릿)\n"
#     "④ 코칭 큐 (3~5개, 4~8글자 명령형)\n"
#     "⑤ 다음 세트 목표 (1문장)\n"
# )
SYSTEM_PROMPT = (
    "너는 한국어 퍼스널 트레이너이다. 사용자는 이미 운동 분석 데이터를 제공했고, "
    "너는 이를 기반으로 **즉시 피드백을 제공**해야 한다.\n\n"

    "🚫 금지사항 (절대 어기지 말 것):\n"
    "- 숫자, 각도, 비율, cm, %, ° 등 **모든 수치 표현 금지**\n"
    "- 수치를 유추하거나 만들어내는 설명 금지\n"
    "- '수치', '각도', '데이터', '정확' 같은 표현 금지\n"
    "- 분석 내용을 그대로 반복하거나 설명 형 문장 금지\n"
    "- '해보세요?', '어떨까요?' 같은 질문형 말투 금지\n\n"

    "✅ 표현 방식 (이 스타일을 강하게 유지):\n"
    "- **코치가 바로 옆에서 말하듯** 부드럽고 단호하게\n"
    "- 감각 기반 표현 사용 (예: '가슴을 부드럽게', '팔을 길게 뻗어', '몸의 중심을 살짝 모아')\n"
    "- 짧고 명확한 문장\n"
    "- *따뜻하지만 확신 있는 톤*\n\n"

    "출력 형식은 아래를 **그대로** 사용하라:\n"
    "① 자세 느낌 요약 (자연스럽게 2문장)\n"
    "② 잘한 점 (1문장)\n"
    "③ 개선할 점 (• 불릿 2~3개)\n"
    "④ 코칭 큐 (• 불릿 3~5개, 4~8글자 명령형)\n"
    "⑤ 다음 세트 목표 (1문장으로 부드럽게)\n"
)



def build_rag_context(user_id: str, user_msg: str, topk: int = 3) -> str:
    """
    로컬 Mongo에서 코사인 유사도 기반 RAG.
    (Atlas 전용 스테이지 사용 안 함)
    """
    qvec = embed(user_msg)
    # 최근 50개만 스캔 (속도/메모리 균형)
    history = list(chat_col.find({"user_id": user_id})
                   .sort("timestamp", -1)
                   .limit(50))
    scored = []
    for h in history:
        vec = safe_get_vec(h)
        if vec is None:
            continue
        sim = cosine_similarity(qvec, vec)
        scored.append((sim, h.get("message", ""), h.get("response", "")))
    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:topk]
    ctx = []
    for _, um, ar in picked:
        ctx.append(f"User: {um}\nAI: {ar}")
    return "\n---\n".join(ctx)

def build_user_prompt(user_msg: str, analysis: Dict[str, Any], user_id: str) -> str:
    # RAG 컨텍스트(있으면 상단 배치하여 우선 반영)
    rag = build_rag_context(user_id, user_msg)

    ex = (analysis or {}).get("detected_exercise") or "미확인 운동"
    stage = (analysis or {}).get("stage") or "단계 정보 없음"
    joints = ((analysis or {}).get("pose_data") or {}).get("joints", {})

    left_elbow  = describe_joint(joints.get("left_elbow_angle"))
    right_elbow = describe_joint(joints.get("right_elbow_angle"))
    left_knee   = describe_joint(joints.get("left_knee_angle"))
    right_knee  = describe_joint(joints.get("right_knee_angle"))
    back_state  = describe_back(joints.get("back_angle"))

    lines = []
    if rag:
        lines.append("[과거 유사 대화]\n" + rag)

    lines.append(
        "[사용자 요청]\n"
        f"{user_msg}\n\n"
        "[분석 상태(참고용) — 출력에 상태 단어를 그대로 쓰지 말고 코칭 문장으로 풀어쓸 것]\n"
        f"- 운동: {ex}\n"
        f"- 단계: {stage}\n"
        f"- 팔: 좌 {left_elbow} / 우 {right_elbow}\n"
        f"- 무릎: 좌 {left_knee} / 우 {right_knee}\n"
        f"- 허리: {back_state}\n\n"
        "위 정보를 기반으로 형식을 정확히 지켜 **코칭 문장**으로만 답변해라. "
        "질문하지 말고 바로 피드백을 제공하라."
    )
    return "\n\n".join(lines)

async def llm_generate(messages: List[Dict[str, str]]) -> str:
    def _run():
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.55,
            top_p=0.9,
            repeat_penalty=1.12,     # ✅ 에코 방지 매우 중요
            max_tokens=600,          # ✅ 끊김 방지
            stop=["<|im_end|>"],    
        )
        return out["choices"][0]["message"]["content"].strip()
    return await asyncio.to_thread(_run)

# ===== Persona 요약 (백그라운드) =====
async def update_persona_background(user_id: str):
    chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(12))
    if not chats:
        return
    text_block = "\n".join([f"User: {c.get('message','')}\nAI: {c.get('response','')}" for c in chats])

    messages = [
        {"role": "system", "content": (
            "아래 최근 대화를 바탕으로 사용자의 운동 습관/목표/통증/선호를 5줄로 요약하라. "
            "숫자 각도 등 세부 수치는 쓰지 마라. 중복 없이 간결하게."
        )},
        {"role": "user", "content": text_block},
    ]
    summary = await llm_generate(messages)
    profile_col.update_one(
        {"user_id": user_id},
        {"$set": {"persona": summary, "updated_at": datetime.now()}},
        upsert=True
    )

# ===== 공통 생성 로직 =====
async def generate_answer(user_id: str, user_msg: str, analysis: Dict[str, Any]) -> str:
    # Persona
    persona_doc = profile_col.find_one({"user_id": user_id})
    persona = persona_doc.get("persona") if persona_doc else ""

    user_block = build_user_prompt(user_msg, analysis, user_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + ("\n[사용자 요약]\n" + persona if persona else "")},
        {"role": "user", "content": user_block},
    ]
    return await llm_generate(messages)

def save_chat(user_id: str, message: str, response: str, embedding: List[float], analysis: Optional[Dict[str, Any]] = None):
    chat_col.insert_one({
        "user_id": user_id,
        "message": message,
        "response": response,
        "embedding": embedding,     # ✅ 항상 384차원으로 저장
        "analysis": analysis or {},
        "timestamp": datetime.now(),
        "embed_model": EMBED_MODEL_NAME,  # ✅ 추후 마이그레이션 대비
        "embed_dim": VECTOR_DIM,
    })

# ===== Endpoints =====
@app.post("/chat")
async def chat_plain(data: ChatInput, background_tasks: BackgroundTasks):
    # 현재 메시지 임베딩
    qvec = embed(data.message)
    # 답변 생성 (분석 없음)
    answer = await generate_answer(data.user_id, data.message, analysis={})
    # 저장
    save_chat(data.user_id, data.message, answer, qvec, analysis={})
    # 페르소나는 3개 이상일 때 주기적으로 갱신
    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)
    return {"answer": answer}

@app.post("/chat_with_analysis")
async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):
    
    qvec = embed(data.message)
    answer = await generate_answer(data.user_id, data.message, analysis=data.analysis)
    save_chat(data.user_id, data.message, answer, qvec, analysis=data.analysis)
    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)
    return {"answer": answer}



# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import asyncio
# import os
# from llama_cpp import Llama

# app = FastAPI()

# # ---------------- MongoDB ----------------
# client = MongoClient("mongodb://localhost:27017")
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]

# # ---------------- Embedding Model ----------------
# embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

# # ---------------- LLM ----------------
# MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=1024,
#     n_threads=8,
#     n_batch=128,
#     logits_all=False,
#     verbose=False,
#     chat_format="chatml"
# )

# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# def cosine_similarity(a, b):
#     a, b = np.array(a), np.array(b)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# async def generate_async(user_msg: str, persona: str, context_text: str):
#     messages = [
#         {"role": "system",
#          "content": (
#              "당신은 개인 맞춤형 건강/운동 상담 코치 AI입니다.\n"
#              "사용자의 지난 대화 내용(persona 요약 + 최근 대화 context)을 참고하여 "
#              "사용자의 상태와 감정, 습관을 기억하고 자연스럽게 이어지는 대화를 하세요."
#          )},
#         {"role": "user",
#          "content": f"[사용자 요약 정보]\n{persona}\n\n[최근 관련 대화]\n{context_text}\n\n[현재 질문]\n{user_msg}"}
#     ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.35,
#             top_p=0.9,
#             max_tokens=240,
#             stop=["</s>", "<|im_end|>"]
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     return await asyncio.to_thread(_run)


# async def update_persona_background(user_id: str):
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(10))
#     text_block = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])

#     messages = [
#         {"role": "system", "content": "최근 대화를 분석하여 사용자의 건강/운동 특징을 5줄로 요약하세요."},
#         {"role": "user", "content": text_block or "(대화없음)"}
#     ]

#     def _run():
#         out = llm.create_chat_completion(messages=messages, temperature=0.2, top_p=0.9, max_tokens=120)
#         return out["choices"][0]["message"]["content"].strip()

#     summary = await asyncio.to_thread(_run)

#     profile_col.update_one(
#         {"user_id": user_id},
#         {"$set": {"persona": summary, "updated_at": datetime.now()}},
#         upsert=True
#     )


# @app.post("/chat")
# async def chat_with_ai(data: ChatInput, background_tasks: BackgroundTasks):

#     # 1) 입력 문장 임베딩
#     emb = embed_model.encode(data.message, normalize_embeddings=True)
#     user_vec = emb.tolist()

#     # 2) 최근 대화 불러오기 + RAG (유사도 상위 3개)
#     history = list(chat_col.find({"user_id": data.user_id}).sort("timestamp", -1).limit(10))

#     contexts = []
#     for h in history:
#         vec = h.get("embedding") or h.get("vector")
#         if not vec:
#             continue
#         if len(vec) != len(user_vec):
#             continue    # ✅ 차원 다르면 skip
#         sim = cosine_similarity(user_vec, vec)
#         contexts.append((sim, h["message"], h.get("response", "")))

#     if contexts:
#         contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:3]
#         context_text = "\n".join([f"User: {m}\nAI: {r}" for _, m, r in contexts])
#     else:
#         context_text = "\n".join([f"User: {h['message']}\nAI: {h['response']}" for h in history[:3]])

#     # 3) Persona 불러오기
#     profile = profile_col.find_one({"user_id": data.user_id})
#     persona = profile["persona"] if profile else "특징 미파악 사용자"

#     # 4) LLM 호출
#     answer = await generate_async(data.message, persona, context_text)

#     # 5) 저장
#     chat_col.insert_one({
#         "user_id": data.user_id,
#         "message": data.message,
#         "response": answer,
#         "embedding": user_vec,
#         "timestamp": datetime.now()
#     })

#     # 6) Persona 업데이트는 백그라운드로
#     if len(history) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)

#     return {"answer": answer, "persona_summary": persona}

