import os 
import joblib
import pandas as pd
import numpy as np
from app.models.sleep_models.sleepSchema import UserInput

MODEL_PATH = "app/models/sleep_models/best_sleep_quality_rf_bundle.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]

# -------------------------------------------------------
# 1. 모델 기반 최적 수면시간 탐색
# -------------------------------------------------------
def find_optimal_sleep(model, scaler, columns, base_input):
    """
    4~10시간을 0.1 단위로 모두 테스트하여
    가장 높은 predicted_quality 점수를 주는 sleep_hours 선택
    """
    hours = np.arange(4.0, 10.0, 0.1)
    scores = []

    for h in hours:
        row = base_input.copy()
        row["sleep_hours"] = h

        df = pd.DataFrame([row])[columns]
        X_scaled = scaler.transform(df)
        y_pred = model.predict(X_scaled)[0]

        scores.append((h, y_pred))

    df_result = pd.DataFrame(scores, columns=["sleep_hours", "predicted_quality"])
    best_row = df_result.loc[df_result["predicted_quality"].idxmax()]  # 최고 수면질

    return df_result, best_row


# -------------------------------------------------------
# 2. 컨디션 레벨 계산
# -------------------------------------------------------
def determine_condition_level(fatigue_score: float) -> str:
    if fatigue_score < 25:
        return "좋음"
    elif fatigue_score < 50:
        return "보통"
    elif fatigue_score < 75:
        return "나쁨"
    else:
        return "최악"


# -------------------------------------------------------
# 3. 피로도 + 최적 수면 시간 예측
# -------------------------------------------------------
def predict_fatigue(data: UserInput):

    # ---------- 모델 입력 ----------
    df = pd.DataFrame([data.model_dump()])[columns]
    X_scaled = scaler.transform(df)
    sleep_quality = float(model.predict(X_scaled)[0])  # 1~4 수면질 점수

    # ---------- 피로도 계산 ----------
    fatigue_score = ((4 - sleep_quality) / 3) * 100  # 0~100

    fatigue_score += data.physical_activity_hours * 2.0
    fatigue_score += data.caffeine_mg * 0.05

    if data.alcohol_consumption == 1:  # boolean → 0 or 1
        fatigue_score += 15.0

    fatigue_score = max(0, min(100, fatigue_score))

    condition = determine_condition_level(fatigue_score)

    # ---------- 모델 기반 최적 수면시간 탐색 ----------
    base_input = data.model_dump()
    _, best_row = find_optimal_sleep(model, scaler, columns, base_input)

    optimal_sleep = round(float(best_row["sleep_hours"]), 2)
    optimal_quality = round(float(best_row["predicted_quality"]), 3)

    # ---------- 추천 구간 생성 ----------
    # 최적값 주변으로 ±0.3h (예: 7.8 → 7.5~8.1)
    recommended_min = round(max(4.0, optimal_sleep - 0.3), 1)
    recommended_max = round(min(10.0, optimal_sleep + 0.3), 1)

    return {
        "predicted_sleep_quality": round(sleep_quality, 3),
        "predicted_fatigue_score": round(fatigue_score, 2),
        "condition_level": condition,
        "optimal_sleep_hours": optimal_sleep,
        "optimal_predicted_quality": optimal_quality,
        "recommended_sleep_range": {
            "min": recommended_min,
            "max": recommended_max
        }
    }
