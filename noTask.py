import os
import json
import multiprocessing
from langchain.llms.base import LLM
import requests

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("HYPERCLOVA_API_KEY")
MODEL_NAME = "clir-emb-dolphin"
URL = f"https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/{MODEL_NAME}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # LangSmith API 키


import uuid

request_id = str(uuid.uuid4())
# --- 헤더 설정 ---
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# --- HyperCLOVA LLM 커스텀 정의 ---
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


llm_final_decider=HyperCLOVA_LLM(
        api_key=API_KEY,
        request_id= request_id,
        system_prompt="""
너는 테스크로 미분류된 질문을 받았을 때 질문을 재생성해서 도구로 해결이 가능하게 하는 모델이야. 감정적인 표현이나 개인적인 불만은 모두 제거하고, 사용자의 실제 의도를 파악하여 정량적이고 도구 기반으로 처리 가능한 질문으로 재구성해줘.

테스크 분류 결과 :{task_result}
원본 질문 : {question}

조건 예시:
- 주가 상승 여부 → RSI ≥ 60 AND 골든크로스 발생
- 주가 하락 여부 → RSI ≤ 40 AND 데드크로스 발생
- 사기 좋은 종목 → RSI > 70 AND 거래량이 20일 평균보다 많음
- 단타 → 최근 10일간 등락률 ≥ 10%
- 예산 기반 → RSI ≥ 50 AND 종가 × 10주 ≤ 예산
- 패턴 탐지 → 예) 3일 연속 양봉이면 적삼병

예시 입력과 출력:
입력: "100만원으로 살 수 있는 좋은 주식 뭐야?"
출력: "2025-07-29 기준 RSI가 50 이상이고 10주 기준 가격이 100만원에 가장 근접한 종목을 알려줘"

입력: "삼성전자 급등하나요?"
출력: "삼성전자의 RSI가 60 이상이면서 골든크로스 발생 여부를 알려줘"

입력: "요새 잘나가는 테마는 무엇이 있을까요?"
출력: "최근 상승률이 높은 업종(예: 2차전지, 반도체 등)에 속한 종목의 RSI와 거래량 기준으로 정리해주세요"

위 조건으로 처리할 수 없는 경우에는 직접 추론을 통해서 정답을 생성해줘. 단, 어떻게 이 결론에 도달했는지를 COT (Chain of Thought) 방식으로 먼저 사고 과정을 설명한 후 마지막에 '결론: ...' 형식으로 최종 질문 재작성 형태를 제시해줘.

재작성된 질문은 반드시 도구로 실행 가능한 형식이어야 하며 최대 2가지의 조건을 가져, 사용 가능한 도구로는 종가/시가/거래량 등의 단순 조회, 이동평균 및 RSI 계산, 골든/데드 크로스 탐지, 거래량 변화 분석, 조건 검색 필터링 등이 있어
""")


def rewrite_question_with_final_decider(task_result, question):
    """
    테스크 분류 결과와 원본 질문을 받아, 도구로 처리 가능한 질문으로 재작성
    """
    prompt = f"""
    테스크 분류 결과 :{task_result}
    원본 질문 : {question}
    """
    return llm_final_decider.invoke(prompt)


if __name__ == "__main__":
    task_result = '[("simple_query", "0"), ("conditional_search", "0"), ("signal_detection", "0")]'

    questions = [
        # "오늘 사서 내일 이득 보고 싶은데 살만한 종목 알려줘",
        # "요즘 잘 나가는 주식 뭐 있어?",
        # "빠르게 수익 낼 수 있는 종목 알려줘",
        # "지금 들어가면 괜찮은 주식은?",
        # "단타로 먹을 주식 하나만 찍어줘",
        # "100만원으로 수익 낼 수 있는 종목 추천해줘",
        # "요즘 핫한 테마 관련 주식 뭐야?",
        # "짧은 기간에 급등할 주식 있을까?",
        # "지금 사도 손해 안 보는 주식 있을까?",
        # "매수하기 좋은 종목 알려줘"

        "옆집 아줌마는 산 주식 다 오르는데 왜 내 건 맨날 떨어지지?",
        "왜 내 주식만 빨간 맛 안 나와?",
        "주식 산 날만 폭락하는 저주에 걸린 것 같아…",
        "왜 다들 테마주 사서 떡상인데 나만 박살나지?",
        "요즘 애들이 사는 주식은 뭔데 그렇게 잘 올라?",
        "한 주식에 몰빵했는데 이거 괜찮은 거지?",
        "부장님이 추천한 주식 샀는데 자꾸 빠지는데 믿어도 돼?",
        "왜 뉴스만 보면 내 주식은 나쁜 얘기만 나오지?",
        "다른 종목은 다 올라서 상한가인데 내 종목은 왜 정지돼있지?",
        "주식이 아니라 인생이 하락 중인 것 같아…"
    ]

    for idx, question in enumerate(questions):
        prompt = f"""
        테스크 분류 결과 :{task_result}
        원본 질문 : {question}
        """
        print(f"\n=== 질문 {idx+1} ===")
        print("❓ 원본 질문:", question)
        result = llm_final_decider.invoke(prompt)
        print("🧠 재작성 결과:\n", result)