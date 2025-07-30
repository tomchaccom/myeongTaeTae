import json
import os
from langchain.llms.base import LLM
import requests
import numpy as np
import pickle
import uuid
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

def build_name_to_ticker_dict(
        kospi_path = "/opt/fastapi-app/fastapi/data/kospi_names.csv", 
        kosdaq_path="/opt/fastapi-app/fastapi/data/kosdaq_names.csv") -> dict:
    kospi_df = pd.read_csv(kospi_path)
    kosdaq_df = pd.read_csv(kosdaq_path)

    kospi_df["Market"] = "KS"
    kosdaq_df["Market"] = "KQ"

    all_df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

    return {
        row["Name"]: f'{str(row["Code"]).zfill(6)}.{row["Market"]}'
        for _, row in all_df.iterrows()
    }

# 사용
name_to_ticker = build_name_to_ticker_dict()

API_KEY = os.getenv("HYPERCLOVA_API_KEY")
MODEL_NAME = "clir-emb-dolphin"
URL = f"https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/{MODEL_NAME}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # LangSmith API 키

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



# system prompt: 종목 줄임말 추출용 (JSON 배열 반환)
ABBR_SYSTEM_PROMPT = """
너는 금융권에서 쓰이는 종목 약어를 JSON 배열 형태로 정확하게 추출하는 역할을 해.
종목명이 줄임말로 나와있는 경우, 그 줄임말을 변환하지 말고 출력에 담아줘야 해
종목명이 정확하게 명시되어 있는 경우에는 [] 형태의 빈 배열을 출력해

예:
질문 : 어제 기준삼전이랑 셀트 중에 뭐가 더 종가가 높아?
답변 : ["삼전", "셀트"]

"""

# HyperCLOVA 인스턴스 생성 (API 키는 Bearer 포함해서 넣기)
llm_abbr = HyperCLOVA_LLM(
    api_key=os.getenv("HYPERCLOVA_API_KEY"),
    request_id=f"abbr-{uuid.uuid4()}",
    system_prompt=ABBR_SYSTEM_PROMPT
)

# 이제 후보 단어를 받아서 임베딩하기 

def get_embedding(text: str) -> np.ndarray | None:
    body = {"text": text}
    try:
        res = requests.post(URL, headers=headers, json=body)
        res.raise_for_status()
        embedding = res.json()["result"]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return None

# print(get_embedding(llm_abbr.invoke("엘지엔솔 상장폐지됐어?")))

# stock_embeddings.pkl에서 종목명 가져오기 
# DenseRetriver(코사인으로 계산하기 )
def load_stock_embeddings(path="/opt/fastapi-app/fastapi/data/stock_embeddings2.pkl"):
    with open(path, "rb") as f:
        df = pickle.load(f)
    stock_names = df["Name"].tolist()
    stock_vecs = np.stack(df["Embedding"].values)
    return stock_names, stock_vecs

from sklearn.metrics.pairwise import cosine_similarity

def get_topk_similar_stocks(query_vec, stock_names, stock_vecs, top_k=5):
    scores = cosine_similarity([query_vec], stock_vecs)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(stock_names[i], scores[i]) for i in top_indices]

llm_expander = HyperCLOVA_LLM(
    api_key=API_KEY,
    request_id=f"expand-{uuid.uuid4()}",
    system_prompt=
       """
        너는 금융권에서 사용되는 주식 은어를 JSON 배열 형태로 정확하게 추출하는 역할을 해.
        주식 은어는 일반인이 이해하기 어려운 표현으로, '적삼병' → '3일간 양봉', '총알' → '투자금'처럼 구체적인 주식 개념을 은어로 표현한 것들이야.

        질문에 은어가 포함되어 있다면 그 은어를 문자열 그대로 배열에 담아줘.
        은어가 없는 경우에는 [] 형태의 빈 배열을 출력해.

        예:
        질문: "삼전전자의 흑삼병 및 떡상 시점 언제야?"
        답변: ["흑삼병", "떡상"]
         """
    
)
llm_slang_rewriter = HyperCLOVA_LLM(
    api_key=API_KEY,
    request_id=f"slang-rewrite-{uuid.uuid4()}",
    system_prompt="""
너는 주식 전문가로서, 사용자의 질문에 포함된 은어 표현을 일반인이 이해할 수 있는 말로 자연스럽게 바꿔주는 역할을 해.

- 입력으로는 원본 질문과 함께 은어 리스트가 주어진다.
- 질문에서 해당 은어들만 정확히 찾아서 바꿔줘야 하며, 다른 부분은 건드리지 마.
- 은어 하나당 하나의 명확한 설명으로 대체하고, 자연스러운 문장 흐름으로 재작성해.
- 결과는 수정된 문장만 출력하고, 다른 부가설명은 하지 마.

예시:
질문: "SK하이 지금 눌림목 구간 같은데, 몰빵해도 될까?"
은어 리스트: ["눌림목", "몰빵"]

출력: "SK하이 지금 단기 조정 구간 같은데, 전액 투자해도 될까?"
"""
)


# sparseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

class SparseRetriever:
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))  # 문자 단위로 변경
        self.stock_tfidf_matrix = self.vectorizer.fit_transform(stock_names)

    def retrieve(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.stock_tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(self.stock_names[i], scores[i]) for i in top_indices]

def normalize_scores(results):
    """
    results: [(name, score), ...] 형태의 리스트
    score들을 0~1 범위로 정규화하여 반환
    """
    scores = [score for _, score in results]
    min_score, max_score = min(scores), max(scores)
    
    if max_score - min_score == 0:
        # 모두 점수가 같으면 1.0으로 통일
        return [(name, 1.0) for name, _ in results]
    
    normalized = [
        (name, (score - min_score) / (max_score - min_score))
        for name, score in results
    ]
    return normalized


def ensemble_results(sparse_results, dense_results, weight_sparse=0.8, weight_dense=0.2):
    all_names = set([name for name, _ in sparse_results] + [name for name, _ in dense_results])
    sparse_dict = dict(sparse_results)
    dense_dict = dict(dense_results)

    # default 값: 가장 낮은 점수 또는 평균 점수 (더 자연스럽게 반영)
    default_sparse = min(sparse_dict.values()) if sparse_dict else 0
    default_dense = min(dense_dict.values()) if dense_dict else 0

    ensemble_scores = {
        name: weight_sparse * sparse_dict.get(name, default_sparse) +
              weight_dense * dense_dict.get(name, default_dense)
        for name in all_names
    }

    return sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)


def rerank_candidates_with_clovax(llm: HyperCLOVA_LLM, question: str, candidates: list[str]) -> str:
    """
    question: 사용자의 원본 질문
    candidates: 앙상블로 뽑은 후보 종목 리스트 (예: ['삼성전자', '셀트리온', '카카오'])
    
    리턴값: 최종적으로 후보 종목 리스트 중 LLM이 판단한 가장 적합한 종목명
    """
    # 후보 리스트를 문자열로 예쁘게 변환
    candidates_str = ", ".join(candidates)
    
    prompt = f"""
    너는 한국 주식 전문가야.
    다음 질문에 가장 관련 있는 종목 하나를 고르고, 그 이유를 간단히 설명해줘.
    
    질문: "{question}"
    후보 종목들 (종목명 기준): {candidates_str}

    가장 적합한 종목명을 출력만 해줘.\
    """
    
    result = llm.invoke(prompt)
    
    # 결과에서 종목명만 깔끔하게 추출하는 후처리 (필요하면 강화 가능)
    chosen_stock = result.strip().split("\n")[0]
    
    return chosen_stock

# 1) 클로바 X LLM 인스턴스 생성
llm_clovax = HyperCLOVA_LLM(
    api_key=API_KEY,
    request_id=f"clovax-{uuid.uuid4()}",
    system_prompt="너는 한국 주식 전문가야. 질문과 후보 종목을 보고 가장 관련 있는 종목을 고르는 역할을 수행해."
)

rewrite_llm = HyperCLOVA_LLM(
    api_key=API_KEY,
    request_id=f"rewrite-{uuid.uuid4()}",
    system_prompt="너는 금융 도메인의 한국어 자연어처리 전문가야. 질문 내의 줄임말을 종목 전체 이름으로 자연스럽게 치환하는 역할을 해."
)
def rewrite_question_with_ticker_code(llm: HyperCLOVA_LLM, original_question: str, tickers: list[str]) -> str:
    """
    original_question: 사용자의 원본 질문 (예: "삼전 요즘 주가 어때?")
    tickers: 리랭킹을 통해 선정된 정식 종목코드 (예: ["005930.KS"])
    
    return: 종목 줄임말를 종목코드로 바꾼 재작성 질문 문자열
    """
    tickers_str = ", ".join(tickers)
    
    prompt = f"""
    아래는 사용자의 원본 질문과, 그 질문에서 언급된 종목 줄임말을 정식 종목코드로 바꾸기 위한 참고 목록이야.

    원본 질문: "{original_question}"
    정식 종목코드 목록: {tickers_str}

    질문에 포함된 종목 줄임말을 아래 목록에 있는 종목코드로 자연스럽게 바꿔서 다시 작성해줘. 문장은 자연스러워야 하고,반드시 종목코드가 모두 반영돼야 해.
    
    반드시 재작성된 질문만 출력해
    
    예시 출력: 켄달스퀘어리츠(365550) 지금 매수해도 괜찮을까요?
    """

    rewritten = llm.invoke(prompt).strip()
    return rewritten


def preprocess_question(user_question: str, stock_names, stock_vecs) -> str:
    """
    사용자 질문을 전처리하는 함수
    
    Args:
        user_question: 사용자 원본 질문
        stock_names: 종목명 리스트
        stock_vecs: 종목 임베딩 벡터
    
    Returns:
        전처리된 질문 문자열
    """
    # 종목 약어 추출
    try:
        llm_response = llm_abbr.invoke(user_question)
        #print("[LLM 약어 추출 응답]", llm_response)
        query_term = json.loads(llm_response)
    except Exception as e:
        print("[LLM 약어 추출 파싱 실패]", e)
        query_term = []
    
    # 추출된 약어가 없으면 바로 은어 변환 단계로 넘어감
    if not query_term:
        print("[약어 추출 없음] 바로 은어 변환 단계로 진행")
        rewritten = user_question
    else:
        answer = []
        for word in query_term:
            query_vec = get_embedding(word)
            dense_results = get_topk_similar_stocks(query_vec, stock_names, stock_vecs, top_k=5)
            top_name, top_score = dense_results[0]
            
            if top_score >= 1.0:
                ticker = name_to_ticker.get(top_name, top_name)
                answer.append(ticker)  # 바로 answer에 추가
                continue
            else:
                sparse_retriever = SparseRetriever(stock_names)
                sparse_results = sparse_retriever.retrieve(word, top_k=5)
                dense_norm = normalize_scores(dense_results)
                sparse_norm = normalize_scores(sparse_results)
                final_results = ensemble_results(sparse_norm, dense_norm)
                candidate_stocks = [name for name, score in final_results]
                best_stock = rerank_candidates_with_clovax(llm_clovax, user_question, candidate_stocks[:5])
                ticker = name_to_ticker.get(best_stock, best_stock)
                answer.append(ticker)
        
        rewritten = rewrite_question_with_ticker_code(rewrite_llm, user_question, answer)
        print(rewritten)
    
    # 은어 변환 단계
    slang_list_raw = (llm_expander.invoke(rewritten))
    try:
        slang_list = json.loads(slang_list_raw)
        if not isinstance(slang_list, list):
            raise ValueError("응답이 리스트 형식이 아님")
    except Exception as e:
        print("[은어 파싱 실패] 예상치 못한 응답:", slang_list_raw)
        slang_list = []
    
    slang_str = ", ".join(slang_list)
    prompt = f"""
    원본 질문: "{rewritten}"
    주식 은어 목록: {slang_str}
    위 질문을 주식 은어를 자연어로 풀어쓰는 문장으로 다시 작성해줘.
    """
    preprocessed_question = llm_slang_rewriter.invoke(prompt)
    return preprocessed_question


if __name__ == "__main__":


    import time
    start_time = time.time() 

    # 종목 벡터 불러오기
    stock_names, stock_vecs = load_stock_embeddings("stock_embeddings2.pkl")

    # 사용자 원본 질문은 별도로 줄임말 추출 전에 받는다고 가정
    user_question = "켄달스퀘어리츠 지금 사도 돼?"

    # 사용자 질문 후보 단어 입력 (줄임말 추출은 별도 함수에서 처리 예정)
    # query_term = json.loads(llm_abbr.invoke(user_question))
    # print(query_term)

    # 추출된 단어의 값으로 검색기를 돌렸는데 값이 1이상이 나오면 프로세스를 진행할 필요가 없다. 
    # 그러면 return 을 하면 될려나?
    
    # answer = []
    # for word in query_term:
    #     # DenseRetriever: 질문 후보 단어 임베딩 후 유사도 검색
    #     query_vec = (get_embedding(word))
    
    #     dense_results = get_topk_similar_stocks(query_vec, stock_names, stock_vecs, top_k=5)

    #     top_name, top_score = dense_results[0]

    #     if top_score >= 1.0:
    #         print("이미 종목 풀네임입니다")
    #         continue
    #     # elif top_score < 0.3:
    #     #     print("휴먼 인터럽트")
    #     #     exit()
    #     else:
    #         # SparseRetriever: 질문 후보 단어로 텍스트 기반 검색
    #         sparse_retriever = SparseRetriever(stock_names)
    #         sparse_results = sparse_retriever.retrieve(word, top_k=5)

    #         # 정규화 수행 (normalize_scores 함수는 별도 정의 필요)
    #         dense_norm = normalize_scores(dense_results)
    #         sparse_norm = normalize_scores(sparse_results)

    #         # 앙상블 점수 계산 (ensemble_results 함수도 별도 정의 필요)
    #         final_results = ensemble_results(sparse_norm, dense_norm)

    #         # 후보 종목 리스트만 추출 (최종 랭킹용)
    #         candidate_stocks = [name for name, score in final_results]

    #         # 리랭킹 수행 (rerank_candidates_with_clovax 함수는 별도 정의됨)
    #         best_stock = rerank_candidates_with_clovax(llm_clovax, user_question, candidate_stocks[:5])

    #         # 결과 출력
    #         # print(f"\n🔍 Dense 결과: {dense_results}")
    #         # print(f"🔍 Sparse 결과: {sparse_results}")
    #         # print(f"🏆 앙상블 Top 종목: {final_results}")
    #         # print(f"🎯 최종 리랭킹된 종목: {best_stock}")
    #         answer.append(best_stock)
    #         # print(answer)

            
        
    #     rewritten = rewrite_question_with_full_stock_names(rewrite_llm, user_question, answer)
    #     # print("✏️ 재작성된 질문:", rewritten)

    #     slang_list_raw = (llm_expander.invoke(rewritten))

    #     try:
    #         slang_list = json.loads(slang_list_raw)
    #     except json.JSONDecodeError:
    #         slang_list = eval(slang_list_raw)

    #     # slang_list가 ["눌림목", "몰빵"] 같은 리스트라면
    #     slang_str = ", ".join(slang_list)

    #     prompt = f"""
    #     원본 질문: "{rewritten}"
    #     주식 은어 목록: {slang_str}

    #     위 질문을 주식 은어를 자연어로 풀어쓰는 문장으로 다시 작성해줘.
    #     """

    #     rewritten_expanded = llm_slang_rewriter.invoke(prompt)
    #     print("✏️ 재작성된 질문:", rewritten_expanded)

    #     end_time = time.time()  # 끝나는 시간 기록
    #     print(f"⏱️ 실행 시간: {end_time - start_time:.4f}초")

    preprocessed_question = preprocess_question(user_question, stock_names, stock_vecs)
    print("✅ 전처리된 질문:", preprocessed_question)

    end_time = time.time()  # 끝나는 시간 기록
    print(f"⏱️ 실행 시간: {end_time - start_time:.4f}초")