# main.py
# 실행: uvicorn main:app --host 0.0.0.0 --port 5000

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

import os, io, math, base64
import numpy as np
import cv2
import av

# MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# --------------------------
# 전역 설정
# --------------------------
app = FastAPI(title="AI Coach Core (FastAPI + MediaPipe + CNN-LSTM)")

# 모델/라벨
cnn_lstm_model = None
LABELS_PATH = "model/labels.txt"
CLASSES: List[str] = []

# 학습 시 사용한 설정과 동일하게 맞추세요
SEQ_LEN = 24
IMG_SIZE = (160, 160)

# 프레임 분석 간격
FRAME_STRIDE = 5  # 5프레임마다 1개 분석

# MediaPipe Pose 전역 재사용
pose_detector = None

# --------------------------
# 커스텀 레이어 (학습과 동일한 이름/로직)
# --------------------------
class TemporalAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)

    def call(self, x):  # x: (B,T,F)
        h = self.dense(x)           # (B,T,U)
        e = self.score(h)           # (B,T,1)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(a * x, axis=1)  # (B,F)
        return context

# --------------------------
# 스키마
# --------------------------
class AnalyzeRequest(BaseModel):
    userId: str
    message: Optional[str] = None     # 텍스트 (그대로 JSON에 실어 반환)
    image: Optional[str] = None       # base64-encoded image
    video: Optional[str] = None       # base64-encoded video

class AnalyzeResponse(BaseModel):
    # 전체 요약
    detected_exercise: Optional[str] = None
    exercise_confidence: Optional[float] = None
    probs: Optional[List[float]] = None

    # 포즈 요약(마지막 프레임)
    stage: Optional[str] = None
    pose_detected: Optional[bool] = None
    pose_data: Optional[Dict[str, Any]] = None

    # 동영상 상세
    frames: Optional[List[Dict[str, Any]]] = None
    total_frames: Optional[int] = None

    # 텍스트 원문(분석 없이 전달)
    message: Optional[str] = None

# --------------------------
# 유틸: 수학/기하
# --------------------------
def angle_3pts(a, b, c) -> float:
    """a,b,c: (x,y); 각도 at b (degree)"""
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    denom = (math.hypot(*ba) * math.hypot(*bc)) + 1e-6
    cosv = (ba[0]*bc[0] + ba[1]*bc[1]) / denom
    cosv = max(-1.0, min(1.0, cosv))
    return round(math.degrees(math.acos(cosv)), 1)

def dist(a, b) -> float:
    return round(math.hypot(a[0]-b[0], a[1]-b[1]), 1)

# --------------------------
# MediaPipe: 포즈 특징 추출 (단일 프레임)
# --------------------------
def extract_pose_features(bgr_img, last_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    res = pose_detector.process(rgb)

    if not res.pose_landmarks:
        return {"pose_detected": False, "pose_data": {}, "stage": "unknown"}

    lm = res.pose_landmarks.landmark

    def xy(i):  # normalize->pixel
        return (lm[i].x * w, lm[i].y * h)

    # 주요 포인트 (.value 중요!)
    L_SH, R_SH = xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value), xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    L_EL, R_EL = xy(mp_pose.PoseLandmark.LEFT_ELBOW.value),    xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    L_WR, R_WR = xy(mp_pose.PoseLandmark.LEFT_WRIST.value),    xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    L_HI, R_HI = xy(mp_pose.PoseLandmark.LEFT_HIP.value),      xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
    L_KN, R_KN = xy(mp_pose.PoseLandmark.LEFT_KNEE.value),     xy(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    L_AN, R_AN = xy(mp_pose.PoseLandmark.LEFT_ANKLE.value),    xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # 중앙점
    SHO = ((L_SH[0]+R_SH[0])/2, (L_SH[1]+R_SH[1])/2)
    HIP = ((L_HI[0]+R_HI[0])/2, (L_HI[1]+R_HI[1])/2)
    KNE = ((L_KN[0]+R_KN[0])/2, (L_KN[1]+R_KN[1])/2)
    ANK = ((L_AN[0]+R_AN[0])/2, (L_AN[1]+R_AN[1])/2)

    # 각도
    elbow_left  = angle_3pts(L_SH, L_EL, L_WR)
    elbow_right = angle_3pts(R_SH, R_EL, R_WR)
    knee_left   = angle_3pts(L_HI, L_KN, L_AN)
    knee_right  = angle_3pts(R_HI, R_KN, R_AN)
    back_angle  = angle_3pts(SHO, HIP, KNE)  # 몸통 기울기 대략

    # 거리/비율
    shoulder_hip = dist(SHO, HIP)
    hip_knee     = dist(HIP, KNE)
    knee_ankle   = dist(KNE, ANK)
    back_to_hip_ratio = round((shoulder_hip / (hip_knee + 1e-6)), 3)

    # 간단 단계(stage)
    mean_knee = (knee_left + knee_right) / 2
    mean_elbow = (elbow_left + elbow_right) / 2
    if mean_knee < 100:
        stage = "down"
    elif mean_elbow < 100:
        stage = "down"
    elif 165 <= back_angle <= 185:
        stage = "hold"
    else:
        stage = "up"

    pose_data = {
        "joints": {
            "left_elbow_angle": elbow_left,
            "right_elbow_angle": elbow_right,
            "left_knee_angle": knee_left,
            "right_knee_angle": knee_right,
            "back_angle": back_angle
        },
        "distances": {
            "shoulder_hip": shoulder_hip,
            "hip_knee": hip_knee,
            "knee_ankle": knee_ankle,
            "back_to_hip_ratio": back_to_hip_ratio
        },
        "keypoints": {
            "shoulder": [round(SHO[0],1), round(SHO[1],1)],
            "hip": [round(HIP[0],1), round(HIP[1],1)],
            "knee": [round(KNE[0],1), round(KNE[1],1)],
            "ankle": [round(ANK[0],1), round(ANK[1],1)],
        }
    }
    return {"pose_detected": True, "pose_data": pose_data, "stage": stage}

# --------------------------
# 이미지 → 시퀀스 추론 (학습 전처리 동일)
# --------------------------
def predict_exercise_from_bgr_seq_model(bgr_img) -> Dict[str, Any]:
    # BGR → RGB → resize
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    x = rgb.astype(np.float32)

    # EfficientNet 전처리: [0,255] 가정
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # 가짜 시퀀스: 동일 프레임을 SEQ_LEN번 복제
    seq = np.stack([x] * SEQ_LEN, axis=0)      # (T,H,W,3)
    inp = np.expand_dims(seq, axis=0)          # (1,T,H,W,3)

    preds = cnn_lstm_model.predict(inp)[0]     # (num_classes,)
    idx = int(np.argmax(preds))
    return {
        "detected_exercise": CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx),
        "exercise_confidence": float(np.max(preds)),
        "probs": preds.tolist()
    }

# --------------------------
# 비디오 → 시퀀스 추론 (균등 샘플링 SEQ_LEN)
# --------------------------
def predict_exercise_from_video_bytes(video_bytes: bytes) -> Dict[str, Any]:
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]

    frames_rgb = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        frames_rgb.append(rgb)
    container.close()

    n = len(frames_rgb)
    if n == 0:
        raise HTTPException(status_code=400, detail="프레임이 없습니다.")

    idxs = np.linspace(0, n-1, num=SEQ_LEN).astype(int)
    samples = []
    for i in idxs:
        rgb = cv2.resize(frames_rgb[i], IMG_SIZE)
        x = tf.keras.applications.efficientnet.preprocess_input(rgb.astype(np.float32))
        samples.append(x)

    seq = np.stack(samples, axis=0)   # (T,H,W,3)
    inp = np.expand_dims(seq, axis=0) # (1,T,H,W,3)

    preds = cnn_lstm_model.predict(inp)[0]
    idx = int(np.argmax(preds))
    return {
        "detected_exercise": CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx),
        "exercise_confidence": float(np.max(preds)),
        "probs": preds.tolist()
    }

# --------------------------
# 오케스트레이션: 이미지 분석
# --------------------------
def analyze_frame(image_bytes: bytes) -> Dict[str, Any]:
    npimg = np.frombuffer(image_bytes, np.uint8)
    bgr   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지 디코딩 실패")

    ex = predict_exercise_from_bgr_seq_model(bgr)
    pose = extract_pose_features(bgr)

    return {
        "detected_exercise": ex["detected_exercise"],
        "exercise_confidence": round(ex["exercise_confidence"], 4),
        "probs": ex["probs"],
        "stage": pose["stage"],
        "pose_detected": pose["pose_detected"],
        "pose_data": pose["pose_data"],
    }

# --------------------------
# 오케스트레이션: 동영상 분석 (프레임별 추론 포함)
# --------------------------
def analyze_video(video_bytes: bytes) -> Dict[str, Any]:
    """
    1) 전체 시퀀스 예측(CNN+LSTM) 1회
    2) FRAME_STRIDE 간격 프레임마다:
       - MediaPipe 포즈(관절/각도/단계)
       - 프레임별 운동 예측(CNN+LSTM, 단일 프레임을 시퀀스로 변환)
    """
    # 1) 전체 시퀀스 예측
    overall = predict_exercise_from_video_bytes(video_bytes)

    frames_data: List[Dict[str, Any]] = []
    try:
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]

        for i, frame in enumerate(container.decode(stream)):
            if i % FRAME_STRIDE != 0:
                continue

            # PyAV frame → numpy RGB → BGR
            rgb = frame.to_ndarray(format="rgb24")
            bgr = rgb[:, :, ::-1].copy()

            # (a) 포즈 특징
            pose = extract_pose_features(bgr)

            # (b) 프레임별 운동 예측
            ex = predict_exercise_from_bgr_seq_model(bgr)

            frames_data.append({
                "frame": i + 1,
                "detected_exercise": ex["detected_exercise"],
                "exercise_confidence": round(ex["exercise_confidence"], 4),
                "probs": ex["probs"],
                "stage": pose["stage"],
                "pose_detected": pose["pose_detected"],
                "pose_data": pose["pose_data"],
            })

        container.close()

    except av.AVError:
        raise HTTPException(status_code=400, detail="PyAV에서 동영상을 읽을 수 없습니다.")

    if not frames_data:
        raise HTTPException(status_code=400, detail="분석 가능한 프레임이 없습니다.")

    # 마지막 프레임 요약(호환 목적)
    last_frame = frames_data[-1]

    return {
        # 전체(클립) 단위 예측
        "detected_exercise": overall["detected_exercise"],
        "exercise_confidence": round(overall["exercise_confidence"], 4),
        "probs": overall["probs"],

        # 프레임별 상세
        "total_frames": len(frames_data),
        "frames": frames_data,

        # 마지막 프레임 요약 필드(기존 응답과 호환)
        "stage": last_frame.get("stage"),
        "pose_detected": last_frame.get("pose_detected"),
        "pose_data": last_frame.get("pose_data"),
    }

# --------------------------
# 라이프사이클: 모델/라벨/포즈 로드 & 종료
# --------------------------
@app.on_event("startup")
def on_startup():
    global cnn_lstm_model, CLASSES, pose_detector

    # 클래스 라벨 로드
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            CLASSES = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # fallback (학습 순서와 다르면 예측 라벨이 달라질 수 있음)
        CLASSES = ["benchpress", "deadlift", "plank", "pushup", "squat"]

    # 모델 로드 (커스텀 레이어 등록)
    cnn_lstm_model = load_model(
        "model/cnn_lstm_exercise_model.keras",
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    # MediaPipe Pose 전역 생성 (재사용)
    pose_detector = mp_pose.Pose(static_image_mode=True)

    print("✅ CNN-LSTM 모델 & Pose 로드 완료. classes =", CLASSES)

@app.on_event("shutdown")
def on_shutdown():
    global pose_detector
    try:
        if pose_detector is not None:
            pose_detector.close()
    except Exception:
        pass

# --------------------------
# 엔드포인트
# --------------------------
@app.get("/")
def root():
    return "OK"

@app.get("/health")
def health():
    return {"status": "AI Core running"}

# --- 추가: JSON 전용 라우트 ---
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_json(payload: AnalyzeRequest):
    try:
        msg = (payload.message or "").strip()

        # 1) 이미지(base64)
        if payload.image:
            image_bytes = base64.b64decode(payload.image)
            result = analyze_frame(image_bytes)
            if msg:
                result["message"] = msg
            # print(result)
            return AnalyzeResponse(**result)

        # 2) 동영상(base64)
        if payload.video:
            video_bytes = base64.b64decode(payload.video)
            result = analyze_video(video_bytes)  # 프레임별 pose/단계/프레임별 예측 포함
            if msg:
                result["message"] = msg
            # print(result)
            return AnalyzeResponse(**result)

        # 3) 텍스트만
        if msg:
            # print(result)
            return AnalyzeResponse(message=msg)

        raise HTTPException(status_code=400, detail="image, video, message 중 하나는 필수입니다.")
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}")

