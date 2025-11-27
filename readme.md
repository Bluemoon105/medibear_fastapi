# 🐻 MediBear - FastAPI

## 팀원 구성
| 이름 | 역할 | Github |
|------|------|---------|
| 🧭 김정규 | FullStack | [@gyu0918](https://github.com/gyu0918) |
| 🌟 유신안 | FullStack | [@shinanyu](https://github.com/shinanyu) |
| 🏗️ 변상용 | FullStack | [@Hayden721](https://github.com/Hayden721) |
| 💫 이승권 | FullStack | [@seoungkwon](https://github.com/seoungkwon) |
| 🎯 임예지 | FullStack | [@Bluemoon105](https://github.com/Bluemoon105) |

## 프로젝트 소개

- 헬스케어와 멘탈케어를 통합한 AI 코팅 웹서비스 구현
- 사용자 개인 맞춤 리포트 및 히스토리 시각화 대시보드 제공
- 프로젝트 기간: 2025.11.03 ~ 2025.11.28

## 기술 스택

### AI / ML / DL
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)

### DB
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)

### 백엔드
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Pydantic](https://img.shields.io/badge/pydantic-%23E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)

### 협업 툴
![Jira](https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)

## 팀원별 구현 기능 상세

### 김정규

### 유신안

### 변상용

### 이상권

### 임예지

## FastAPI – Sleep LLM 분석 서비스 개발 내역
### 1. 수면 분석 LLM API 개발
- Spring Boot로부터 전달받은 수면 데이터를 입력받아 분석 수행
- Groq/OpenAI 모델 기반 LLM 호출
- 수면 시간 · 카페인 · 활동량 등을 기반으로 맞춤형 코칭 메시지 생성
- 분석 결과를 JSON 형태로 표준화하여 반환

### 2. Prompt Engineering 기반 분석 템플릿 구성
- 일관적인 분석 결과 생성을 위한 Prompt 구조 설계
- 특수 문자/한글 깨짐 문제 해결을 위한 후처리 적용
- "수면 점수 · 피로도 · 개선 팁" 중심 응답 구조 최적화

### 3. 모델 결과 구조화 및 표준화
- LLM 응답 파싱(JSON parsing) 안정화
- Spring Boot에서 바로 사용할 수 있는 형태로 변환

## 모델 학습 성능 비교 및 응답 품질 모니터링
### 1. MLflow 기반 수면 예측 모델 성능 비교
- 수면 피로도 예측·권장 수면 시간 추천을 위해 다양한 ML 모델 실험(RandomForest, XGBoost, LightGBM 등)
- MLflow Tracking으로 R²·MAE·RMSE 등을 자동 기록하고 실험 이력 관리
- MLflow UI를 이용해 모델 성능을 시각적으로 비교하여 최적 모델 선정
- 선정된 모델을 `.pkl`로 FastAPI에서 로드하여 실시간 추론 API 제공

### 2. LangSmith를 활용한 LLM 응답 속도 및 품질 모니터링
- Groq/OpenAI 모델 호출 시 LangSmith Trace 활성화
- 요청별 Latency, Token 사용량, Prompt/Response 로그 자동 기록
- Prompt Engineering 개선 시 응답 속도·안정성 변화 비교 분석
- 수면 분석 API의 전체 LLM 품질·성능 모니터링 체계 구축

