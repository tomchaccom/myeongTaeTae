import os
from langchain.llms.base import LLM
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("HYPERCLOVA_API_KEY")
import uuid
request_id = str(uuid.uuid4())

PROMPTS = {
    "simple_query": """
    너는 금융 도메인에서 "단순조회" 질문만을 판별하는 전문 판단기야.

    단순조회란, 아래와 같은 질문 유형을 의미해:
    - 특정 날짜에 종목의 시가, 종가, 고가, 저가, 등락률, 거래량 등을 직접적으로 묻는 질문
    - 특정 날짜의 전체 시장(KOSPI, KOSDAQ 등) 정보 조회 (예: 전체 거래대금, 종목 수)
    - 상승/하락 종목 수, 거래량이 많은 종목, 가장 비싼 종목 등을 특정 날짜 기준으로 묻는 단순 통계 질문

    주의:
    - 조건(예: 거래량이 n% 이상, 가격이 a~b만원 등)이 붙으면 단순조회가 아님
    - 패턴 감지나 평균 비교, 복잡한 조건이 포함되면 단순조회가 아님

    예시:
    - "7월 25일에 삼성전자의 종가는 얼마였어?" → 1
    - "3월 22일에 거래량 많은 종목 5개 알려줘" → 1
    - "20일 이동평균보다 종가가 높은 종목 알려줘" → 0

    질문: {question}

    위 조건을 만족하는 경우에만 "1"을 출력하고, 그 외는 "0"을 출력해.
    """,

    "conditional_search": """
    너는 금융 도메인에서 "조건 검색" 질문만을 판별하는 전문 판단기야.

    조건 검색이란, 아래와 같은 질문 유형을 의미해:
    - 특정 날짜에 종목이 특정 조건을 만족하는 경우 (예: 거래량이 전일 대비 n% 이상, 가격이 특정 구간 내)
    - 등락률, 거래량, 종가 등의 수치 조건이 비교나 범위 조건으로 사용된 경우
    - 조건이 2개 이상 결합된 복합 필터링도 포함됨

    주요 패턴:
    - "전날 대비","n% 이상", "n만원 이하", "n만주 초과", "범위 안" 등의 조건문
    - KOSPI/KOSDAQ 조건이 함께 제시되어도 포함됨

    주의:
    - 반드시 질문에 거래량, 종가, 등락율에 대한 정량적 수치가 포함되어야 함(n%, n만원), 그렇지 않은 질문(종목의 갯수를 물어보는 질문)은 단순조회
    - 골드크로스, RSI, 볼린저밴드, 이동평균 등 기술 분석 시그널은 조건검색이 아님 → 시그널 감지로 분류해야 함

    예시:
    - "8월 1일에 거래량이 10만주 이상인 종목은?" → 1
    - "1월 7일에 종가가 5만원 이상 10만원 이하인 종목 알려줘" → 1
    - "20일 평균보다 거래량이 많은 종목" → 1
    - "삼성전자 종가 얼마야?" → 0

    질문: {question}

    위 조건을 만족하는 경우에만 "1"를 출력하고, 그 외는 "0"을 출력해.

    """,

    "signal_detection": """
    너는 금융 도메인에서 "시그널 감지" 질문만을 판별하는 전문 판단기야.

    시그널 감지란, 아래와 같은 기술적 분석 기반의 현상이나 조건을 탐지하려는 질문을 의미해:
    - 골든크로스, 데드크로스 발생 여부나 횟수
    - 거래량이 20일 평균 대비 +-n% 이상 증가)
    - RSI, MACD, 볼린저밴드, 이동평균 돌파 등 기술 지표 기반 조건
    - 시가/종가의 이동평균 이탈, 특정 기준선 돌파

    주요 키워드:
    - 골든크로스, 데드크로스, RSI, 볼린저, 이동평균, 상대 강도 지수

    주의:
    - 조건이 붙더라도 주요 키워드에 포함된 기술적 지표 기반이 아니면 조건검색으로 분류해야 함

    예시:
    - "2025년 6월 1일 부터 한달간 골든크로스 발생했어?" → 1
    - "7월 25일에 RSI가 70 이상인 종목 알려줘" → 1
    - "볼린저 밴드 상단 터치한 종목" → 1
    - "LG전자 2025년 4월 25일 종가 알려줘" → 0

    질문: {question}

    위 조건을 만족하는 경우에만 "1"을 출력하고, 그 외는 "0"을 출력해.
    """
}

class HyperCLOVA_LLM(LLM):
    api_key: str
    request_id: str
    system_prompt: str
    endpoint: str = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    @property
    def _llm_type(self) -> str:
        return "hyperclova"
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        headers = {
            'Authorization': self.api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "topP": 0.8,
            "temperature": 0,
            "maxTokens": 256
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["result"]["message"]["content"].strip('"').strip()

def create_llm(task_name: str, request_id: str) -> HyperCLOVA_LLM:
    return HyperCLOVA_LLM(
        api_key=API_KEY,
        request_id=request_id,
        system_prompt=PROMPTS[task_name]
    )

def classify_task(task_name: str, question: str) -> tuple[str, str]:
    llm = create_llm(task_name, f"{task_name}-request")
    prompt = PROMPTS[task_name].format(question=question)
    try:
        result = llm.invoke(prompt)
    except Exception as e:
        result = "0"
    return (task_name, result.strip()) 

# 2. LLM 인스턴스 생성 (Clova X 기준)
llm_final_decider=HyperCLOVA_LLM(
        api_key=API_KEY,
        request_id= request_id,
        system_prompt="""

        너는 입력으로 들어오는 결과를 보고 1이 나온 키의 값을 반환하면 돼

        입력 형식:[
        ('simple_query', '0'), 
        ('conditional_search', '1'), 
        ('signal_detection', '주어진 질문은 거래량과 관련된 내용이지만, 주요 키워드에 포함되지 않았으므로 조건 검색으로 분류됩니다. 따라서 이 질문에는 "0"을 출력합니다.')]
        ]
        
        각 튜플의 두 번째 문자열에는 설명이 포함될 수 있어.  
        하지만 반드시 `"0"`, `"1"` 중 하나가 포함되어 있어.  

        출력 형식:[
            ('simple_query', '0'),
            ('conditional_search', '1'),
            ('signal_detection', '1')
        ]

        주의 사항:
        절대 문장을 분석하지 말고, 숫자만 추출해.
        출력 형식을 반드시 지켜서 출력해
        설명을 요약하지 말고, 변형하지 말고, 무시해.
                
    """
    )
