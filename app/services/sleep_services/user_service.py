from sqlalchemy import text
from datetime import date
from app.services.sleep_services.db_service import get_engine

engine = get_engine()

# 회원 기본 정보 조회
def get_user_info(member_no: int):
    """PostgreSQL에서 회원 기본 정보 조회"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT name, gender, birth_date 
                FROM member_tb 
                WHERE member_no = :member_no
            """),
            {"member_no": member_no}
        ).mappings().first()

    if not result:
        return {
            "name": "알 수 없음",
            "gender": "비공개",
            "age": "비공개",
        }

    # 생년월일 → 나이 계산
    birth = result["birth_date"]
    today = date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

    return {
        "name": result["name"],
        "gender": result["gender"],
        "age": age
    }


# 하루 활동 정보 조회
def get_daily_activity(member_no: int):
    """PostgreSQL에서 특정 회원의 최신 하루 활동 데이터 조회"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    sleep_hours,
                    predicted_fatigue_score,
                    recommended_sleep_range,
                    predicted_sleep_quality,
                    caffeine_mg,
                    alcohol_consumption,
                    physical_activity_hours
                FROM daily_activities_tb
                WHERE member_no = :member_no
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"member_no": member_no}
        ).mappings().first()

    return result or {
        "sleep_hours": "N/A",
        "predicted_fatigue_score": "N/A",
        "recommended_sleep_range": "N/A",
        "predicted_sleep_quality": "N/A",
        "caffeine_mg": "N/A",
        "alcohol_consumption": "N/A",
        "physical_activity_hours": "N/A",
    }


# 주간 활동 정보 조회
def get_weekly_activity(member_no: int):
    """PostgreSQL에서 특정 회원의 최근 7일간 활동 데이터 조회"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT *
                FROM daily_activities_tb
                WHERE member_no = :member_no
                ORDER BY date DESC
                LIMIT 7
            """),
            {"member_no": member_no}
        ).mappings().all()

    return result
