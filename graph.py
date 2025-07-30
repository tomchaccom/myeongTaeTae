from typing import List, Optional, Dict, Any, Tuple, Literal, Annotated
from langgraph.graph import END, StateGraph
from typing import TypedDict 

# === 추가: 전처리 및 테스크 분류 모듈 import ===
from rag_stock_pipeline import preprocess_question, HyperCLOVA_LLM, load_stock_embeddings
from noTask import rewrite_question_with_final_decider
from task_classification import classify_task
import os

# === LLM 인스턴스 및 임베딩 준비 ===

API_KEY = os.getenv("HYPERCLOVA_API_KEY")
stock_names, stock_vecs = load_stock_embeddings("/Users/myeongsung/미래 에셋/active/stock_embeddings2.pkl")


class WorkflowState(TypedDict):
    original_question: Annotated[str, "사용자가 입력한 원본 질문"]
    preprocessed_question: Annotated[str, "종목 약어, 은어 등이 치환된 질문"]
    is_ambiguous: Annotated[bool, "질문이 모호한지 여부"]
    rewrite_candidates: Annotated[List[str], "질문 재작성 후보 리스트"]
    rewritten_question: Annotated[Optional[str], "재작성된 질문"]
    clarification_required: Annotated[bool, "사용자에게 질문 의도 확인 필요 여부"]
    confirmed_question: Annotated[Optional[str], "사용자 확인을 거친 최종 질문"]
    task_candidates: Annotated[List[Tuple[str, str]], "LLM이 판단한 후보 테스크 리스트 (이름, 코드)"]
    task_type: Annotated[Optional[Literal["simple_query", "conditional_search", "signal_detection"]], "확정된 테스크 타입"]
    task_code: Annotated[Optional[int], "확정된 테스크 코드"]
    tool_calls: Annotated[List[Dict[str, Any]], "호출해야 할 도구 리스트 및 인자"]
    no_task_result : Annotated[Any, "미분류 테스크 질문 재작성 결과"]
    result: Annotated[Any, "도구 호출 결과"]

# ✅ 2. 각 노드 함수 정의 (더미 구현 → 실제 구현)
def user_input_node(state: WorkflowState) -> WorkflowState:
    return state

def preprocessing_node(state: WorkflowState) -> WorkflowState:
    # 실제 전처리 함수 호출
    preprocessed = preprocess_question(
        state["original_question"],
        stock_names,
        stock_vecs
    )
    state["preprocessed_question"] = preprocessed
    return state

def ambiguity_check_node(state: WorkflowState) -> WorkflowState:
    # LLM이 파라미터 추출 가능 여부 판단 (임시)
    state["is_ambiguous"] = False
    state["clarification_required"] = False
    return state

def question_rewrite_node(state: WorkflowState) -> WorkflowState:
    state["rewritten_question"] = state["preprocessed_question"] + " (재작성됨)"
    return state

def clarification_node(state: WorkflowState) -> WorkflowState: # 사용자에게 재작성된 질문을 확인 받는 노드
    state["confirmed_question"] = state.get("rewritten_question", state.get("preprocessed_question"))
    return state

def task_classification_node(state: WorkflowState) -> WorkflowState:
    # 실제 테스크 분류 함수 호출
    candidates = []
    for task in ["simple_query", "conditional_search", "signal_detection"]:
        task_name, result = classify_task(task, state["preprocessed_question"])
        candidates.append((task_name, result))
    state["task_candidates"] = candidates
    # 가장 높은 점수(혹은 1)인 task를 task_type에 할당 (예시)
    for task_name, result in candidates:
        if result == "1":
            state["task_type"] = task_name
            state["task_code"] = 1
            break
    else:
        state["task_type"] = None
        state["task_code"] = 0
    return state

def decide_task_node(state: WorkflowState) -> str:
    task_result = state.get("task_candidates", [])
    final_task = None
    for task_name, result in task_result:
        if result == "1":
            final_task = task_name
            break
    if final_task is None:
        final_task = "no_task"
        state["task_code"] = 0
    else:
        state["task_code"] = 1
    state["task_type"] = final_task  # ✅ 그래프 분기용 확정 테스크
    return final_task  # ✅ 이 값이 LangGraph 분기 기준이 됨



def no_task_node(state: WorkflowState) -> WorkflowState:
    # task_type이 None이면 미분류로 간주하여 질문 재작성
    print("-----")
    if state.get("task_type") is None:
        task_result = state.get("task_candidates", [])
        question = state.get("preprocessed_question", "")
        rewritten = rewrite_question_with_final_decider(task_result, question)
        state["rewritten_question"] = rewritten
        state["no_task_result"] = {
            "message": "미분류 질문 재작성 완료",
            "rewritten": rewritten
        }
    else:
        state["no_task_result"] = {
            "message": "task_type이 None이 아니므로 no_task_node는 실행되지 않음",
            "task_type": state["task_type"]
        }
    
    return state

def agent_node(state: WorkflowState) -> WorkflowState:
    state["result"] = {"message": "도구 실행 완료"}
    return state

def result_node(state: WorkflowState) -> WorkflowState:
    print("[결과]", state.get("result"))
    return state

# ✅ 3. 조건 분기 함수
def should_rewrite(state: WorkflowState) -> str:
    return "rewrite" if state.get("is_ambiguous") else "skip"

def needs_clarification(state: WorkflowState) -> str:
    return "clarify" if state.get("clarification_required") else "no_clarify"

def decide_task_node(state: WorkflowState) -> str:
    task_result = state.get("task_candidates", [])
    final_task = None
    for task_name, result in task_result:
        if result == "1":
            final_task = task_name
            break
    if final_task is None:
        final_task = "no_task"
    state["task_type"] = final_task  # ✅ 그래프 분기용 확정 테스크
    return final_task  # ✅ 이 값이 LangGraph 분기 기준이 됨

# ✅ 4. LangGraph 그래프 구성
builder = StateGraph(WorkflowState)

builder.add_node("user_input", user_input_node)
builder.add_node("preprocess", preprocessing_node)
builder.add_node("ambiguity_check", ambiguity_check_node)
builder.add_node("question_rewrite", question_rewrite_node)
builder.add_node("clarification", clarification_node)
builder.add_node("task_classify", task_classification_node)
builder.add_node("agent", agent_node)
builder.add_node("result", result_node)

builder.add_node("no_task", no_task_node)

builder.set_entry_point("user_input")
builder.add_edge("user_input", "preprocess")
builder.add_edge("preprocess", "ambiguity_check")
builder.add_conditional_edges("ambiguity_check", should_rewrite, {
    "rewrite": "question_rewrite",
    "skip": "task_classify"
})
builder.add_conditional_edges("question_rewrite", needs_clarification, {
    "clarify": "clarification",
    "no_clarify": "task_classify"
})

builder.add_edge("clarification", "task_classify")
builder.add_node("decide_task", decide_task_node)
# 수정: no_task 분기 → no_task 노드로
builder.add_conditional_edges("task_classify", decide_task_node, {
    "simple_query": "agent",
    "conditional_search": "agent",
    "signal_detection": "agent",
    "no_task": "no_task"
})

builder.add_edge("no_task", "agent")
builder.add_edge("agent", "result")

builder.add_edge("result", END)

workflow = builder.compile()



questions = [
        # "엔씨 양봉 전환 가능할까?", # 0
        # "2025-05-14에 거래량이 전날대비 300% 이상 증가한 종목을 모두 보여줘", # 2
        # "2025-01-20에 RSI가 70 이상인 과매수 종목을 알려줘", # 3
        # "동부건설우의 2024-11-06 시가은?", # 1
        # "넷마블 음봉 연속 며칠째야?", # 0
        # "2024-08-30에 등락률이 +5% 이상이면서 거래량이 전날대비 300% 이상 증가한 종목을 모두 보여줘", # 2
        # "2025-01-13에 RSI가 80 이상인 과매수 종목을 알려줘", # 3
        # "2025-06-23에 상승한 종목은 몇 개인가?", # 1
        # "삼성전자 오를까?", # 0
        # "2024-11-22에 거래량이 2000만주 이상인 종목을 모두 보여줘", # 2
        # "2024-09-02에 MACD가 시그널을 상향 돌파한 종목을 알려줘", # 3
        # "금양그린파워의 2024-08-08 종가는?", # 1
        "엘지생건 지금 사도 돼?", # 0
        # "2025-05-27에 등락률이 -3% 이하인 종목을 모두 보여줘", # 2
        ]


for question in questions:
    final_state = workflow.invoke({"original_question": question})

    print("전처리 결과:", final_state["preprocessed_question"])
    print("테스크 후보:", final_state["task_candidates"])
    print("최종 테스크:", final_state["task_type"], final_state["task_code"])
    # print("미분류 테스크 :",final_state["no_task_result"])
    if final_state["task_code"] == 0:
        print("미분류 테스크 :",final_state["no_task_result"]["rewritten"])
    print("결과",final_state["result"])
    
