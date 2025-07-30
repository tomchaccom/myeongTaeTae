from fastapi import FastAPI, Query
from graph import workflow  # graph.py와 같은 디렉터리에 있어야 함

app = FastAPI()

@app.get("/ask")
def ask_question(question: str = Query(..., description="사용자 질문")):
    try:
        state = workflow.invoke({"original_question": question})

        result_text = f"""
[질문] {question}
[전처리 질문] {state.get('preprocessed_question')}
[테스크 후보] {state.get('task_candidates')}
[최종 질문 (COT)] {state.get('confirmed_question') or state.get('rewritten_question') or '없음'}
[최종 테스크] {state.get('task_type')} (코드 {state.get('task_code')})
"""
        if state.get("task_code") == 0:
            result_text += f"[미분류 재작성] {state['no_task_result']['rewritten']}\n"

        result_text += f"[최종 결과] {state.get('result')}"
        return {"result": result_text.strip()}

    except Exception as e:
        return {"error": str(e)}