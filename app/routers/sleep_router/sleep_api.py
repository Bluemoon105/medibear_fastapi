import pandas as pd
from fastapi import APIRouter
from app.models.sleep_models.sleepSchema import UserInput
from app.services.sleep_services.sleep_service import (
    predict_fatigue,   # 모델 기반 + 최적 수면시간 포함
    find_optimal_sleep,  # 새로 추가된 모델 기반 최적 수면시간 함수
    model,
    scaler,
    columns,
)

router = APIRouter(prefix="/sleep", tags=["sleep"])


# -------------------------------------------------------
# 1) 피로도 + 최적 수면시간 반환
# -------------------------------------------------------
@router.post("/predict-fatigue")
async def predict_fatigue_endpoint(data: UserInput):
    """
    사용자 입력 데이터를 받아
    - 피로도 점수
    - 컨디션 레벨
    - 최적 수면시간(모델 기반)
    - 예측된 수면 질
    을 반환합니다.
    """
    return predict_fatigue(data)



# -------------------------------------------------------
# 2) 추천 수면시간 범위 API (모델 기반)
# -------------------------------------------------------
@router.post("/recommend")
def recommend_by_model(data: UserInput):
    """
    모델 기반으로 4~10시간 구간에서 최적 수면시간을 탐색하고,
    min/max 추천 범위를 계산한다.
    """

    # DataFrame 변환
    df = pd.DataFrame([data.model_dump()])[columns]
    X_scaled = scaler.transform(df)

    # 현재 sleep_quality 예측
    base_quality = float(model.predict(X_scaled)[0])

    # 모델 기반 최적 수면시간 탐색
    base_input = data.model_dump()
    df_result, best_row = find_optimal_sleep(model, scaler, columns, base_input)

    optimal_sleep = round(float(best_row["sleep_hours"]), 2)
    optimal_quality = round(float(best_row["predicted_quality"]), 3)

    # 추천 범위를 ±0.3시간으로 설정
    min_h = round(max(4.0, optimal_sleep - 0.3), 1)
    max_h = round(min(10.0, optimal_sleep + 0.3), 1)

    return {
        "predicted_sleep_quality": round(base_quality, 3),
        "optimal_sleep_hours": optimal_sleep,
        "optimal_predicted_quality": optimal_quality,
        "recommended_sleep_range": f"{min_h} ~ {max_h} 시간",
        "min": min_h,
        "max": max_h
    }
