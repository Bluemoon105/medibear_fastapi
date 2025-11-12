from fastapi import APIRouter, UploadFile, File
from typing import Literal
from PIL import Image
import io
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ophtha.llm.graph import workflow
from app.services.ophtha.vision.cataract import analyze_cataract
from app.services.ophtha.vision.glaucoma import analyze_glaucoma
from app.schema.ophtha.ophtha_schema import ChatIn, ChatOut, GraphState, AnalyzeOut

router = APIRouter()

@router.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    text = body.text.strip()
    # 사용자가 "안할래" 등으로 이미지 분석 거절 시
    if any(k in text for k in ["안할래", "괜찮아", "필요없어", "안할게", "안할께"]):
        state = {"session_id": body.session_id}
        out = workflow.invoke(state, config={"configurable": {"thread_id": body.session_id}}, start_at="declined")
        return ChatOut(session_id=body.session_id, category="declined", advice=out.get("advice", ""))
    
    # 기본 LLM + LangGraph 흐름
    state: GraphState = {"session_id": body.session_id, "input": text}
    out = workflow.invoke(state, config={"configurable": {"thread_id": body.session_id}})
    
    return ChatOut(
		session_id=body.session_id,
		category=str(out.get("category", "general")),
		advice=out.get("advice", "")
	)
    
# 질병 분류
@router.post("/upload/{disease}", response_model=AnalyzeOut)
async def upload(disease: Literal["cataract", "glaucoma"],
                file: UploadFile = File(...),
                session_id: str = "default"):
	image_bytes = await file.read()
	if disease == "cataract":
		image_summary, advice = analyze_cataract(image_bytes)
	else:
		image_summary, advice = analyze_glaucoma(image_bytes)
	return AnalyzeOut(
		session_id = session_id,
		disease = disease,
		image_summary = image_summary,
		advice = advice
    
	)
 
