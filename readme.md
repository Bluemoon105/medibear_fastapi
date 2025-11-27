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
- 피로도 예측 및 수면 시간 추천
    - MLFlow로 각 모델의 성능 확인 및 최적의 모델 선정
    - ML의 XGBoost 하이퍼파라미터 튜닝 모델로 학습
- AI 수면 코치 챗봇
    - LangGraph 기반으로 3개의 노드로 분리 
    - 일상 대화 노드
        - 사용자와 수면에 관한 대화를 할 수 있는 수면 코치
    - 일간 리포트 노드
        - 사용자가 입력한 하루 활동량 기반으로 만든 리포트
    - 주간 리포트 노드
        - 사용자가 입력한 최근 7일간 데이터의 평균값으로 만든 리포트
    - LangSmith로 응답 속도 확인하며 프롬프트 개선
