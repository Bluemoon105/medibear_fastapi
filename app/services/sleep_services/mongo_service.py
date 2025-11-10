from pymongo import MongoClient, DESCENDING, ASCENDING
from datetime import datetime
from config.sleep_config import settings
import certifi

client = MongoClient(
    settings.MONGO_URL,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=20000,
    )
db = client[settings.MONGO_DB_NAME]
chat_collection = db["chat_history"]

#대화 내용 저장
def save_chat(user_id: int, user_message: str, bot_response: str, chat_type: str = "general"):
    """대화 내용을 MongoDB에 저장"""
    chat_doc = {
        "user_id": user_id,
        "chat_type": chat_type,   # 'general' 또는 'sleep'
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow(),
    }
    chat_collection.insert_one(chat_doc)
    
#최근 10개 데이터 불러오기    
def get_user_chats(user_id: int, limit: int = 10):
    """가장 최근 10개의 대화 가져오되, 오래된 → 최신 순으로 정렬"""
    
    # 최신순으로 limit개 가져오기
    chats = list(
        chat_collection.find({"user_id": user_id})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )

    if not chats:
        return []

    #최신순으로 가져왔으니, 표시할 땐 반대로 (오래된 → 최신 순)
    chats.reverse()

    #ObjectId 문자열 변환
    for chat in chats:
        if "_id" in chat:
            chat["_id"] = str(chat["_id"])

    return chats