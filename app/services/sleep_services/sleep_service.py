import os
import joblib
import pandas as pd
import numpy as np
from app.models.sleep_models.sleepSchema import UserInput

# 모델 로드
MODEL_PATH = "app/models/sleep_models/best_sleep_quality_rf_bundle.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]

# 1) 수면 시간 추천 (규칙 기반)
def rule_based_sleep_recommendation(
    sleep_hours: float,
    physical_activity_hours: float,
    caffeine_mg: float,
    alcohol_consumption: float,   # 0 or 1
    fatigue_score: float,
) -> tuple[float, float]:
    """
    사용자 상태(수면/활동/카페인/알코올/피로도)를 종합하여
    권장 수면시간 범위를 추천
    """

    # 기본 추천 범위
    min_optimal, max_optimal = 7.0, 7.5

    # ① 수면 충분 + 활동 적음 + 카페인 적음 + 술 X
    if (
        sleep_hours >= 7.5 
        and physical_activity_hours < 1.0 
        and caffeine_mg < 100 
        and alcohol_consumption == 0
    ):
        min_optimal, max_optimal = 6.5, 7.0

    # ② 평균적 생활 → 정상 범위
    elif (
        6.5 <= sleep_hours <= 7.5
        and 1.0 <= physical_activity_hours <= 2.0
        and 100 <= caffeine_mg <= 250
    ):
        min_optimal, max_optimal = 7.5, 8.0

    # ③ 활동 많음 / 카페인 높음 / 술 마심 / 수면 부족
    elif (
        sleep_hours < 6.5
        or physical_activity_hours > 2.0
        or caffeine_mg > 250
        or alcohol_consumption == 1
    ):
        min_optimal, max_optimal = 8.0, 8.5

    # ④ 피로도 높으면 추가 수면 필요
    if fatigue_score >= 60:
        min_optimal += 0.5
        max_optimal += 0.5

    # 최대 9시간 제한
    max_optimal = min(max_optimal, 9.0)

    return round(min_optimal, 1), round(max_optimal, 1)


# 2) 피로도 + 컨디션 계산
def predict_fatigue(data: UserInput):
    # 입력 변환
    df = pd.DataFrame([data.model_dump()])[columns]
    X_scaled = scaler.transform(df)

    # 모델 기반 수면 질 예측 (1 ~ 4)
    sleep_quality = float(model.predict(X_scaled)[0])

    # 피로도 점수 (0 ~ 100) — 최신 공식 적용
    fatigue_score = ((4 - sleep_quality) / 3) * 100

    # 보정값: 활동·카페인·알코올 영향 추가
    fatigue_score += data.physical_activity_hours * 2.0
    fatigue_score += data.caffeine_mg * 0.05
    fatigue_score += data.alcohol_consumption * 5.0  # alcohol = 0 or 1

    # 범위 제한
    fatigue_score = float(np.clip(fatigue_score, 0, 100))

    # 컨디션 단계
    if fatigue_score < 25:
        condition = "좋음"
    elif fatigue_score < 50:
        condition = "보통"
    elif fatigue_score < 75:
        condition = "나쁨"
    else:
        condition = "최악"

    # 결과 반환
    return {
        "predicted_sleep_quality": round(sleep_quality, 3),
        "predicted_fatigue_score": round(fatigue_score, 1),
        "condition_level": condition
    }
