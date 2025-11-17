import pandas as pd
from fastapi import APIRouter
from app.models.sleep_models.sleepSchema import UserInput
from app.services.sleep_services.sleep_service import (
    predict_fatigue,         # 피로도 + 컨디션 계산
    find_optimal_sleep,      # 모델 기반 최적 수면시간 계산
    model,
    scaler,
    columns,
)

router = APIRouter(prefix="/sleep", tags=["sleep"])


# -------------------------------------------------------
# 1) 피로도 예측 + 최적 수면시간 + 추천 범위
# -------------------------------------------------------
@router.post("/predict-fatigue")
async def predict_fatigue_endpoint(data: UserInput):
    """
    사용자 입력 데이터를 받아 피로도, 컨디션, 최적 수면시간을 계산 후 반환
    (Spring Boot 저장 필드 이름에 정확히 맞춤)
    """
    # 1. 기본 피로도/컨디션 계산
    base = predict_fatigue(data)

    # 2. 모델 기반 최적 수면시간 탐색
    base_input = data.model_dump()
    df_result, best_row = find_optimal_sleep(model, scaler, columns, base_input)

    optimal_sleep = round(float(best_row["sleep_hours"]), 1)

    # ±0.3 시간 추천 범위
    min_h = round(max(4.0, optimal_sleep - 0.3), 1)
    max_h = round(min(10.0, optimal_sleep + 0.3), 1)
    recommended_range = f"{min_h} ~ {max_h} 시간"

    # 3. Spring Boot가 저장하는 필드 이름 그대로 반환
    return {
        "predicted_sleep_quality": base["predicted_sleep_quality"],
        "predicted_fatigue_score": base["predicted_fatigue_score"],
        "condition_level": base["condition_level"],
        "recommended_sleep_range": recommended_range
    }



# -------------------------------------------------------
# 2) 수면 추천만 따로 호출하는 API (필요 시)
# -------------------------------------------------------
@router.post("/recommend")
def recommend_rule_based(data: UserInput):
    """
    모델 기반 최적 수면시간 ±0.3 범위만 반환
    (반환 이름은 기존과 동일)
    """
    # 모델 기반 탐색
    base_input = data.model_dump()
    df_result, best_row = find_optimal_sleep(model, scaler, columns, base_input)

    optimal_sleep = round(float(best_row["sleep_hours"]), 1)

    min_h = round(max(4.0, optimal_sleep - 0.3), 1)
    max_h = round(min(10.0, optimal_sleep + 0.3), 1)

    return {
        "predicted_sleep_quality": None,     # Spring은 이 값 안 씀 (단순 형식 유지용)
        "predicted_fatigue_score": None,
        "recommended_sleep_range": f"{min_h} ~ {max_h} 시간",
    }
