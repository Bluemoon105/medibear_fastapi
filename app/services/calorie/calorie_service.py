import os
import joblib
import pandas as pd
import json
from dotenv import load_dotenv
from groq import Groq


APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(APP_DIR, "models", "calorie")

MODEL_PATH = os.path.join(MODEL_DIR, "calorie_model_hgbr.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_dict.json")

# 모델 로드
model = joblib.load(MODEL_PATH)

with open(ENCODER_PATH, 'r') as f:
    encoder_dict = json.load(f)

# 2) 환경변수 로드
load_dotenv()

# 3) Groq client 생성
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 칼로리 예측 함수
def predict_calories(duration_minutes: float, 
    weight_kg: float, activity_type: str, bmi: float, gender: str):

    # 입력 데이터프레임 생성
    data = pd.DataFrame({
        "duration_minutes": [duration_minutes],
        "weight_kg": [weight_kg],
        "activity_type": [activity_type],
        "bmi": [bmi],
        "gender": [gender]
    })
    
    # 범주형 인코딩: activity_type + gender
    for col in ["activity_type", "gender"]:

        input_val = data[col][0]

        # 정상적으로 존재하는 카테고리라면 그대로 매핑
        if input_val in encoder_dict[col]:
            data[col] = [encoder_dict[col][input_val]]
        else:
            print(f"'{input_val}'은 학습 데이터에 없음 → Unknown 처리")
            data[col] = [encoder_dict[col]["Unknown"]]

    prediction = model.predict(data)[0]

    return round(float(prediction), 2)


# 🔥 Groq LLM 분석 함수
def llm_anaylze_calorie(request):
    
    user = request.member
    latest = request.latest
    logs = request.logs
    # DB에서 가지고온 운동타입, 시간, 소모한 칼로리, 몸무게 데이터를 텍스트로 변환
    log_text = "\n".join(
        f"{l.activityType}/{l.durationMinutes}분/{l.caloriesBurned}kcal/{l.weightKg}kg"
        for l in logs
    )

    prompt = f"""
        너는 최고의 헬스 코치야. 아래는 사용자의 운동 기록이니까 이를 바탕으로 분석해줘.
    단, 운동 기록(raw data)은 분석에는 사용하되 **최종 답변에는 그대로 노출하지 마라.**
    
    사용자 정보:
    - 이름: {user.name}
    - 성별: {user.gender}
    - 최신 키: {latest.heightCm} cm
    - 최신 몸무게: {latest.weightKg} kg
    - 최신 BMI: {latest.bmi}
    
        아래는 운동 기록 데이터(LLM 분석용):
    ---
    {log_text}
    ---
    위 데이터는 분석에만 사용하고 답변에 노출하지 마라.


    응답:
    1) 운동 패턴 요약 
    2) 칼로리 소모 추세 
    3) 15일 후 예상 몸무게 변화 (칼로리 소모량 기반)
    - 제공된 운동 기록으로 계산된 '운동 소모 칼로리'만을 기반으로 단순 예측할 것
    - 나이, BMR, TDEE, 섭취 칼로리 등 제공되지 않은 정보는 사용하지 말 것
    - 기초대사량 공식(Mifflin-St Jeor 등)을 절대 사용하지 말 것
    - 단, 운동 소모 칼로리만으로 추정한 단순 계산은 반드시 수행할 것
    - 식사 정보가 없어 정확한 체중 변화는 계산할 수 없다는 단서를 반드시 포함할 것
   
    한국어로 작성해줘.

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
