import os
import joblib
import pandas as pd
import json
from dotenv import load_dotenv
from groq import Groq


APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(APP_DIR, "models", "calorie")

MODEL_PATH = os.path.join(MODEL_DIR, "calorie_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_dict.json")

# 1) 모델 로드
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ENCODER_PATH, 'r') as f:
    encoder_dict = json.load(f)

# 2) 환경변수 로드
load_dotenv()

# 3) Groq client 생성
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# 🔥 칼로리 예측 함수
# -----------------------------
def predict_calories(duration_minutes: float, weight_kg: float, activity_type: str, bmi: float):
    data = pd.DataFrame({
        'duration_minutes': [duration_minutes],
        'weight_kg': [weight_kg],
        'activity_type': [activity_type],
        'bmi': [bmi]
    })

    # activity_type 인코딩
    if activity_type in encoder_dict['activity_type']:
        data['activity_type'] = [encoder_dict['activity_type'][activity_type]]
    else:
        data['activity_type'] = [encoder_dict['activity_type']['Unkown']]

    # 스케일링
    data_scaled = scaler.transform(data)

    # 예측
    calorie_prediction = model.predict(data_scaled)[0]
    return round(float(calorie_prediction), 2)


# -----------------------------
# 🔥 Groq LLM 분석 함수
# -----------------------------
def llm_anaylze_calorie(logs):
    # compact log formatting
    log_text = "\n".join(
        f"{l.activityType}/{int(l.caloriesBurned/8)}분/{l.caloriesBurned}kcal/{l.weightKg}kg"
        for l in logs
    )

    prompt = f"""
    데이터:
    {log_text}

    응답:
    1) 운동 패턴 요약 
    2) 칼로리 소모 추세 
    3) 15일 후, 30일 후 몸무게 예측
    한국어로 자세히 작성.

    그리고 마지막에 다음 형식으로 "요약:"을 포함해 1줄로 요약해주세요:
    요약: ~~~
    """

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-120b",
        temperature=0.3
    )

    advice = response.choices[0].message.content.strip()

    # "요약:" 부분만 파싱
    summary = None
    if "요약:" in advice:
        summary = advice.split("요약:")[-1].strip()
    else:
        summary = advice[:200]  # fallback

    return {
        "prompt": prompt,
        "advice": advice,
        "summary": summary
    }
