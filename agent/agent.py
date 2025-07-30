#!/usr/bin/env python3
"""
주식 분석 AI 에이전트 (개선된 9단계 플로우 버전)
LangGraph StateGraph를 사용한 체계적인 9단계 프로세스 구현

플로우:
1. 의도 파악
2. 텍스트 계획 생성
2b. 텍스트 계획을 리스트로 파싱
3. 계획 검증
4. 계획 수정 (검증 실패 시)
5. 계획 분해 (리스트 파싱)
6. 계획 구체화 (작업단위로 실행)
7. 계획별 파라미터 세팅 및 도구 검증 (반복 - 성공할 때까지)
8. 작업 실행 (큐 기반)
9. 결과 출력
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_naver import ChatClovaX
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from tool_utils import TOOL_MAP, get_openai_function_definitions, get_tool_info_string, get_tools_info_string, safe_json_parse
import prompts
# ===== 상태 정의 =====
class ToolValidationFailed(Exception):
    pass


class StockAgentState(TypedDict):
    """개선된 에이전트 상태"""
    # 입력
    user_input: str
    
    # 1단계: 의도 파악
    intention_analysis: str
    
    # 2단계: 텍스트 계획
    text_plan: str
    
    # 2b단계: 텍스트 계획을 리스트로 파싱
    parsed_plan_list: List[str]  # 파싱된 계획 리스트
    
    # 3단계: 계획 검증
    plan_validation_result: str
    validation_feedback: str
    
    # 4단계: 계획 수정 (검증 실패 시)
    revised_text_plan: str
    
    # 5단계: 계획 분해 (리스트 파싱)
    step_by_step_plan: List[Dict[str, Any]]  # 단계별 세부 계획
    
    # 6단계: 계획 구체화 (작업단위로 실행)
    detailed_plan: List[Dict[str, Any]]
    
    # 7단계: 계획별 파라미터 세팅 및 도구 검증
    parameter_definition_results: List[Dict[str, Any]]
    tool_validation_results: List[Dict[str, Any]]
    task_queue: List[Dict[str, Any]]  # 실행 큐
    

    
    # 9단계: 작업 실행
    task_execution_results: List[Dict[str, Any]]
    execution_summary: Dict[str, Any]
    
    # 10단계: 결과 출력
    final_answer: str
    
    # 공통
    error_message: str
    current_stage: str
    revision_count: int


# ===== LLM 설정 =====

def get_llm():
    """ChatClovaX LLM 인스턴스 생성"""
    return ChatClovaX(
        model='HCX-005',
        temperature=0.0,
        top_p=0.1,
        max_tokens=4096,
        api_key=os.getenv("CLOVASTUDIO_API_KEY"),
    )

# ===== 10단계 노드 구현 =====

def step1_intention_analysis(state: StockAgentState):
    """1단계: 의도 파악"""
    print("🎯 1단계: 의도 파악 중...")
    
    user_input = state["user_input"]
    llm = get_llm()
    
    prompt = prompts.get_intention_analysis_prompt(user_input)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print(response.content)
    
    return {
        "intention_analysis": response.content,
        "current_stage": "의도_파악_완료"
    }

def step2_text_planning(state: StockAgentState):
    """2단계: 텍스트 계획 생성"""
    print("📝 2단계: 텍스트 계획 생성 중...")
    
    intention_analysis = state["intention_analysis"]
    user_input = state["user_input"]
    llm = get_llm()
    
    prompt = prompts.get_text_planning_prompt(
        intention_analysis=intention_analysis,
        user_input=user_input
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    
    # 텍스트 계획에서 개별 작업들 파싱
    text_plan = response.content
    
    return {
        "text_plan": text_plan,
        "current_stage": "텍스트_계획_완료"
    }

def step2b_parse_plan_to_list(state: StockAgentState):
    """2b단계: 텍스트 계획을 리스트로 파싱"""
    print("📋 2b단계: 텍스트 계획을 리스트로 파싱 중...")
    
    text_plan = state["text_plan"]
    llm = get_llm()
    
    # 텍스트 계획을 리스트로 파싱하는 프롬프트
    prompt = f"""
텍스트 계획을 단계별 리스트로 파싱해주세요.

## 텍스트 계획
{text_plan}

## 요구사항
1. 텍스트 계획을 개별 단계로 분리하세요
2. 각 단계는 명확하고 실행 가능해야 합니다
3. JSON 배열 형태로 반환하세요

## 출력 형식
```json
[
  "1단계: 첫 번째 작업 설명",
  "2단계: 두 번째 작업 설명",
  "3단계: 세 번째 작업 설명"
]
```

파싱된 계획 리스트를 반환해주세요.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    
    # 안전한 JSON 파싱 사용
    parsed_plan_list = safe_json_parse(response.content)
    
    print(f"  ✅ 계획 파싱 완료: {len(parsed_plan_list)}개 단계")
    
    return {
        "parsed_plan_list": parsed_plan_list,
        "current_stage": "계획_파싱_완료"
    }

def step5_plan_decomposition(state: StockAgentState):
    """5단계: 계획 분해 (리스트 파싱)"""
    print("🔧 5단계: 계획 분해 중...")
    
    # 수정된 계획이 있으면 사용, 없으면 원본 사용
    text_plan = state.get("revised_text_plan", state["text_plan"])
    llm = get_llm()
    
    # 계획 분해 프롬프트
    prompt = f"""
텍스트 계획을 실행 가능한 단계별 작업으로 분해해주세요.

## 텍스트 계획
{text_plan}

## 분해 요구사항
1. 각 단계를 구체적이고 실행 가능한 작업으로 분해하세요
2. 각 작업에 적절한 도구를 지정하세요
3. 작업 간의 의존성을 고려하세요
4. JSON 형태로 구조화된 계획을 반환하세요. 문자열은 반드시 큰따옴표를 사용하세요.
5. 도구가 없는 경우 'N/A'로 표시하세요
6. 값의 타입에 유의하세요. 빈 값은 빈 문자열로 표시하세요.

출력 예시:
```json
[
  {{
    "task_number": 2,
    "task_description": "2025-01-01 종목 거래이력 데이터 조회",
    "tool": {{
      "name": "get_stock_price_history",
      "description": "특정 종목의 특정 날짜 거래이력 데이터를 조회합니다."
    }},
    "dependencies": [1]
  }},
  {{
    "task_number": 3,
    "task_description": "등락률 계산식 추론",
    "tool": {{
      "name": "N/A",
      "description": "도구가 없는 작업입니다."
    }},
    "dependencies": []
  }}
]
```

출력:
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    
    # 안전한 JSON 파싱 사용
    step_by_step_plan = safe_json_parse(response.content)
    
    print(f"  ✅ 계획 분해 완료: {len(step_by_step_plan)}개 작업")
    
    return {
        "step_by_step_plan": step_by_step_plan,
        "current_stage": "계획_분해_완료"
    }

def step6_plan_elaboration(state: StockAgentState):
    """6단계: 계획 구체화 (작업단위로 실행)"""
    print("🔧 6단계: 계획 구체화 중...")
    
    step_by_step_plan = state["step_by_step_plan"]
    llm = get_llm()
    
    detailed_plan = []
    
    print(f"  📋 총 {len(step_by_step_plan)}개 작업 구체화 중...")
    
    for task_info in step_by_step_plan:
        task_number = task_info["task_number"]
        task_description = task_info["task_description"]
        tool_info = task_info.get("tool", {})
        
        print(f"    📋 작업 {task_number}: 구체화 중...")
        
        # 작업 구체화 프롬프트
        prompt = f"""
작업을 더 구체적으로 정의해주세요.

## 작업 정보
작업 번호: {task_number}
작업 설명: {task_description}
도구: {tool_info.get('name', 'N/A') if isinstance(tool_info, dict) else str(tool_info)}

## 구체화 요구사항
1. 작업의 구체적인 실행 방법을 정의하세요
2. 필요한 입력과 예상 출력을 명시하세요
3. 작업의 우선순위와 의존성을 고려하세요
4. 값의 타입에 유의하세요.
5. 다른 설명 없이 결과만 출력해주세요.
6. 출력 예시의 포멧과 동일하게 출력해주세요.

## 출력 예시
```json
{{
  "task_number": 2,
  "task_description": "주식 데이터 조회",
  "tool": {{
    "name": "get_stock_price_history",
    "description": "특정 종목의 특정 날짜 거래이력 데이터를 조회합니다."
  }},
  "dependencies": [1]
}}
```
출력 결과:
"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        elaborated_task = safe_json_parse(response.content)
        
        detailed_plan.append(elaborated_task)
        print(f"    ✅ 작업 {task_number}: 구체화 완료")
    
    print(f"  📊 계획 구체화 완료: {len(detailed_plan)}개 작업")
    
    return {
        "detailed_plan": detailed_plan,
        "current_stage": "계획_구체화_완료"
    }

def step3_plan_validation(state: StockAgentState):
    """3단계: 계획 검증"""
    print("🔍 3단계: 계획 검증 중...")
    
    user_input = state["user_input"]
    parsed_plan_list = state["parsed_plan_list"]
    intention_analysis = state["intention_analysis"]
    llm = get_llm()
    
    # 계획 검증 프롬프트
    prompt = f"""
사용자의 요청과 파싱된 계획을 검증해주세요.
도구를 활용하면 25년도 데이터까지 조회 가능합니다.

## 사용자 요청
{user_input}

## 의도 분석
{intention_analysis}

## 파싱된 계획 리스트
{parsed_plan_list}

## 제공 도구 목록
- get_current_date: 오늘 날짜를 반환합니다.
- calculate: 수학 계산을 수행합니다. 사칙연산과 기본 수학 함수를 지원합니다.
- filter_stocks_by_indicator_auto: 주식 데이터를 필터링하여 조건에 맞는 종목을 반환합니다.
- get_stock_price_history: 특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.

## 검증 기준
1. 계획이 사용자 요청을 완전히 만족하는가?
2. 각 단계가 명확하고 실행 가능한가?
3. 계획이 논리적 순서로 구성되어 있는가?
4. 누락된 중요한 단계가 있는가?

## 응답 형식
YES 또는 NO로 시작하고, 그 뒤에 검증 결과와 피드백을 작성하세요.

출력 예시:
YES - 계획이 적절하고 실행 가능합니다.
NO - 계획에 문제가 있습니다. [구체적인 문제점과 개선 방향]
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    
    validation_result = response.content.strip()
    
    # 검증 결과 분석
    if "YES" in validation_result:
        print("  ✅ 계획 검증 통과")
        return {
            "plan_validation_result": "YES",
            "validation_feedback": validation_result,
            "current_stage": "계획_검증_통과"
        }
    else:
        print("  ❌ 계획 검증 실패")
        return {
            "plan_validation_result": "NO",
            "validation_feedback": validation_result,
            "current_stage": "계획_검증_실패"
        }

def step4_plan_revision(state: StockAgentState):
    """4단계: 계획 수정"""
    print("✏️ 4단계: 계획 수정 중...")
    
    intention_analysis = state["intention_analysis"]
    validation_feedback = state["validation_feedback"]
    user_input = state["user_input"]
    
    llm = get_llm()
    
    # 계획 수정 프롬프트
    prompt = f"""
검증 피드백을 바탕으로 계획을 수정해주세요.

## 사용자 요청
{user_input}

## 에이전트가 사용 가능한한 도구 목록
- get_current_date: 오늘 날짜를 반환합니다.
- calculate: 수학 계산을 수행합니다. 사칙연산과 기본 수학 함수를 지원합니다.
- filter_stocks_by_indicator_auto: 주식 데이터를 필터링하여 조건에 맞는 종목을 반환합니다.
- get_stock_price_history: 특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.

## 의도 분석
{intention_analysis}

## 검증 피드백
{validation_feedback}

## 수정 요구사항
1. 검증 피드백에서 지적된 문제점을 해결하세요
2. 더 명확하고 실행 가능한 계획으로 개선하세요
3. 누락된 단계가 있다면 추가하세요
4. 논리적 순서로 재구성하세요

수정된 텍스트 계획을 작성해주세요.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    
    return {
        "text_plan": response.content,
        "current_stage": "계획_수정_완료"
    }



def step7_parameter_setting(state: StockAgentState):
    """7단계: 계획별 파라미터 세팅 및 도구 검증"""
    print("⚙️ 7단계: 계획별 파라미터 세팅 및 도구 검증 중...")
    
    detailed_plan = state["detailed_plan"]
    intention_analysis = state["intention_analysis"]
    previous_results = state.get("previous_results", {})
    llm = get_llm()
    
    validation_queue = detailed_plan.copy()
    parameter_definition_results = []
    tool_validation_results = []
    task_queue = []  # 실행 큐
    
    print(f"  📋 총 {len(detailed_plan)}개 작업의 파라미터 설정 및 검증 중...")
    
    while validation_queue:
        task_info = validation_queue.pop(0)
        task_number = task_info["task_number"]
        task_description = task_info["task_description"]
        tool_info = task_info.get("tool", {})
        tool_name = tool_info.get("name", "") if isinstance(tool_info, dict) else str(tool_info)
        
        print(f"    ⚙️ 작업 {task_number}: 파라미터 설정 및 검증 중...")
        
        # 도구가 "N/A"이거나 없는 경우 처리
        if not tool_name or tool_name == "N/A" or tool_name == "no-tool":
            print(f"    ⚠️ 작업 {task_number}: 도구 없음 - 큐에 추가")
            param_result = {
                "task_number": task_number,
                "tool_name": "N/A",
                "parameters": {},
                "note": "도구가 없는 작업입니다."
            }
            validation_info = {
                "task_number": task_number,
                "tool_name": "N/A",
                "validation_result": "YES",
                "validation_feedback": "도구가 없는 작업이므로 검증을 건너뜁니다.",
                "parameters": {}
            }
            
            parameter_definition_results.append(param_result)
            tool_validation_results.append(validation_info)
            
            # 큐에 추가 (도구 없는 작업도 큐에 포함)
            task_queue.append({
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": "N/A",
                "parameters": {},
                "validation_result": "YES"
            })
            continue
        
        # 도구 정보 가져오기
        tool_info_string = get_tool_info_string(tool_name) if tool_name else "도구 정보 없음"
        
        # 파라미터 설정 프롬프트
        prompt = f"""
작업에 필요한 파라미터를 설정해주세요.
## 이전 작업 결과
{previous_results}

## 작업 정보
작업 번호: {task_number}
작업 설명: {task_description}
도구: {tool_name}

## 의도 분석
{intention_analysis}

## 도구 정보
{tool_info_string}

## 파라미터 설정 요구사항
1. 도구에 필요한 모든 파라미터를 정의하세요
2. 각 파라미터의 타입과 설명을 명시하세요
3. 기본값이 필요한 경우 설정하세요
4. 필수 파라미터와 선택적 파라미터를 구분하세요

## 출력 예시
```json
{{
  "task_number": 2,
  "tool_name": "get_stock_price_history",
  "parameters": {{
    "stock_code": "005930",
    "date": "2024-01-01"
  }}
}}
```

파라미터 설정 결과를 반환해주세요.
"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        param_result = safe_json_parse(response.content)
        
        # task_number 추가
        if isinstance(param_result, dict):
            param_result["task_number"] = task_number
        
        parameter_definition_results.append(param_result)
        parameters = param_result.get("parameters", {})
        
        # 도구 검증 수행
        print(f"    🛠️ 작업 {task_number}: 도구 검증 중...")
        
        # 도구 검증 프롬프트
        validation_prompt = f"""
도구 사용법이 올바른지 검증해주세요.

## 작업 정보
작업 번호: {task_number}
작업 설명: {task_description}
도구: {tool_name}

## 설정된 파라미터
{parameters}

## 도구 정보
{tool_info_string}

## 검증 기준
1. 도구가 작업에 적합한가?
2. 파라미터가 올바르게 설정되었는가?
3. 필수 파라미터가 누락되지 않았는가?
4. 파라미터 타입이 올바른가?

## 응답 형식
YES 또는 NO로 시작하고, 그 뒤에 검증 결과와 피드백을 작성하세요.

출력 예시:
YES - 도구 사용법이 올바릅니다.
NO - 도구 사용법에 문제가 있습니다. [구체적인 문제점과 개선 방향]
"""
        
        validation_response = llm.invoke([HumanMessage(content=validation_prompt)])
        validation_result = validation_response.content.strip()
        
        validation_info = {
            "task_number": task_number,
            "tool_name": tool_name,
            "validation_result": "YES" if "YES" in validation_result else "NO",
            "validation_feedback": validation_result,
            "parameters": parameters
        }
        
        tool_validation_results.append(validation_info)
        
        # 검증 결과에 따라 큐에 추가
        if "YES" in validation_result:
            print(f"    ✅ 작업 {task_number}: 검증 통과 - 실행 큐에 추가")
            task_queue.append({
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": tool_name,
                "parameters": parameters,
                "validation_result": "YES"
            })
        else:
            print(f"    ❌ 작업 {task_number}: 검증 실패 - 검증 큐에 추가")
            print(f"    📝 피드백: {validation_result}")
            # 검증 실패한 작업을 validation_queue에 추가
            validation_queue.append({
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": tool_name,
                "parameters": parameters,
                "validation_feedback": validation_result,
                "revision_attempts": 0  # 수정 시도 횟수 초기화
            })
        
        print(f"    ✅ 작업 {task_number}: 파라미터 설정 및 검증 완료")
    
    print(f"  📊 파라미터 설정 및 검증 완료: {len(parameter_definition_results)}개 작업")
    print(f"  📋 실행 큐에 추가된 작업: {len(task_queue)}개")
    
   
    print(f"  ✅ 모든 작업 검증 통과 - 실행 단계로 이동")
    return {
        "task_queue": task_queue,
        "current_stage": "파라미터_설정_및_검증_완료"
    }





def step9_tool_execution(state: StockAgentState):
    """9단계: 작업 실행 (큐 기반)"""
    print("🚀 9단계: 작업 실행 중...")
    
    task_queue = state["task_queue"]
    user_input = state["user_input"]
    

    
    task_execution_results = []
    total_start_time = time.time()
    
    print(f"  📋 큐에서 {len(task_queue)}개 작업 실행 예정")
    print(f"  ⏰ 실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 작업 간 컨텍스트 전달을 위한 변수
    previous_results = {}
    
    for i, task_info in enumerate(task_queue):
        task_number = task_info["task_number"]
        task_description = task_info["task_description"]
        tool_name = task_info["tool_name"]
        parameters = task_info["parameters"]
        validation_result = task_info["validation_result"]
        
        print(f"  🔧 작업 {task_number}: {task_description} (도구: {tool_name})")
        
        # 도구가 "N/A"이거나 없는 경우 처리
        if not tool_name or tool_name == "N/A" or tool_name == "no-tool":
            print(f"    ⚠️ 작업 {task_number}: 도구 없음 - 작업 건너뛰기")
            execution_result = {
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": "N/A",
                "parameters": {},
                "execution_status": "도구_없음",
                "result": f"작업 '{task_description}'은 도구 없이 수행되는 작업입니다.",
                "error": None,
                "execution_time": datetime.now().isoformat()
            }
            task_execution_results.append(execution_result)
            continue
        
        # 검증 결과에 따른 처리
        if validation_result == "YES":
            print(f"    ✅ 검증 통과 - 작업 실행")
            
            # 작업 실행 결과 초기화
            execution_result = {
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": tool_name,
                "parameters": parameters,
                "execution_status": "성공",
                "result": None,
                "error": None,
                "execution_time": datetime.now().isoformat()
            }
            
            try:
                # 도구 실행
                tool = TOOL_MAP.get(tool_name)
                if tool:
                    print(f"    📝 파라미터: {parameters}")
                    result = tool.invoke(parameters)
                    print(f"    📊 결과: {result}")
                    execution_result["result"] = result
                    print(f"    ✅ 작업 {task_number}: 실행 성공")
                else:
                    execution_result["execution_status"] = "실패"
                    execution_result["error"] = f"도구 '{tool_name}'를 찾을 수 없습니다"
                    print(f"    ❌ 작업 {task_number}: 도구를 찾을 수 없음")
                    
            except Exception as e:
                execution_result["execution_status"] = "실패"
                execution_result["error"] = str(e)
                print(f"    ❌ 작업 {task_number}: 실행 실패 - {str(e)}")
            
            task_execution_results.append(execution_result)
            
            # 다음 작업에 컨텍스트 전달
            if execution_result["result"]:
                previous_results[f"task_{task_number}"] = execution_result["result"]
                
        else:
            print(f"    ❌ 검증 실패 - 작업 건너뛰기")
            
            # 검증 실패한 작업도 결과에 추가
            execution_result = {
                "task_number": task_number,
                "task_description": task_description,
                "tool_name": tool_name,
                "parameters": parameters,
                "execution_status": "검증_실패",
                "result": None,
                "error": "검증 단계에서 실패한 작업입니다.",
                "execution_time": datetime.now().isoformat()
            }
            task_execution_results.append(execution_result)
    
    # 전체 실행 시간 계산
    total_end_time = time.time()
    total_duration = round(total_end_time - total_start_time, 3)
    
    print(f"  ⏰ 실행 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📊 총 실행 시간: {total_duration}초")
    
    # 성공/실패 통계
    success_count = sum(1 for result in task_execution_results if result["execution_status"] == "성공")
    tool_na_count = sum(1 for result in task_execution_results if result["execution_status"] == "도구_없음")
    failure_count = len(task_execution_results) - success_count - tool_na_count
    
    print(f"  📈 실행 결과: 성공 {success_count}개, 도구없음 {tool_na_count}개, 실패 {failure_count}개")
    
    return {
        "task_execution_results": task_execution_results,
        "current_stage": "작업_실행_완료",
        "execution_summary": {
            "total_tasks": len(task_queue),
            "successful_tasks": success_count,
            "tool_na_tasks": tool_na_count,
            "failed_tasks": failure_count,
            "total_duration_seconds": total_duration,
            "completion_time": datetime.now().isoformat()
        }
    }

def step10_result_output(state: StockAgentState):
    """10단계: 결과 출력"""
    print("📊 10단계: 결과 출력 중...")
    
    intention_analysis = state["intention_analysis"]
    task_execution_results = state["task_execution_results"]
    user_input = state["user_input"]
    llm = get_llm()
    
    # 결과 출력 프롬프트
    prompt = f"""
사용자의 요청에 대한 최종 결과를 정리해주세요.

## 사용자 요청
{user_input}

## 의도 분석
{intention_analysis}

## 작업 실행 결과
{task_execution_results}

## 요구사항
1. 사용자의 원래 질문에 대한 명확한 답변을 제공하세요
2. 실행된 작업들의 결과를 종합하여 의미 있는 정보를 제공하세요
3. 실패한 작업이 있다면 그 이유와 대안을 제시하세요
4. 사용자가 이해하기 쉽게 정리해주세요

최종 결과를 작성해주세요.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "final_answer": response.content,
        "current_stage": "완료"
    }

def should_continue(state: StockAgentState):
    """다음 단계 결정 - 새로운 플로우에 맞게 수정"""
    current_stage = state.get("current_stage", "")
    
    if current_stage == "의도_파악_완료":
        return "step2_text_planning"
    elif current_stage == "텍스트_계획_완료":
        return "step2b_parse_plan_to_list"
    elif current_stage == "계획_파싱_완료":
        return "step3_plan_validation"
    elif current_stage == "계획_검증_통과":
        return "step5_plan_decomposition"
    elif current_stage == "계획_검증_실패":
        return "step4_plan_revision"
    elif current_stage == "계획_수정_완료":
        return "step2b_parse_plan_to_list"  # 수정된 계획을 다시 파싱
    elif current_stage == "계획_분해_완료":
        return "step6_plan_elaboration"
    elif current_stage == "계획_구체화_완료":
        return "step7_parameter_setting"
    elif current_stage == "파라미터_설정_및_검증_완료":
        return "step9_tool_execution"
    elif current_stage == "작업_실행_완료":
        return "step10_result_output"
    elif current_stage == "오류":
        return END
    else:
        return END

def create_stock_agent():
    """개선된 9단계 에이전트 생성"""
    
    # StateGraph 생성
    workflow = StateGraph(StockAgentState)
    
    # 노드 추가
    workflow.add_node("step1_intention_analysis", step1_intention_analysis)
    workflow.add_node("step2_text_planning", step2_text_planning)
    workflow.add_node("step2b_parse_plan_to_list", step2b_parse_plan_to_list)
    workflow.add_node("step3_plan_validation", step3_plan_validation)
    workflow.add_node("step4_plan_revision", step4_plan_revision)
    workflow.add_node("step5_plan_decomposition", step5_plan_decomposition)
    workflow.add_node("step6_plan_elaboration", step6_plan_elaboration)
    workflow.add_node("step7_parameter_setting", step7_parameter_setting)
    workflow.add_node("step9_tool_execution", step9_tool_execution)
    workflow.add_node("step10_result_output", step10_result_output)
    
    # 시작점 설정
    workflow.set_entry_point("step1_intention_analysis")
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "step1_intention_analysis",
        should_continue,
        {
            "step2_text_planning": "step2_text_planning",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step2_text_planning",
        should_continue,
        {
            "step2b_parse_plan_to_list": "step2b_parse_plan_to_list",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step2b_parse_plan_to_list",
        should_continue,
        {
            "step3_plan_validation": "step3_plan_validation",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step3_plan_validation",
        should_continue,
        {
            "step5_plan_decomposition": "step5_plan_decomposition",
            "step4_plan_revision": "step4_plan_revision",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step4_plan_revision",
        should_continue,
        {
            "step2b_parse_plan_to_list": "step2b_parse_plan_to_list",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step5_plan_decomposition",
        should_continue,
        {
            "step6_plan_elaboration": "step6_plan_elaboration",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step6_plan_elaboration",
        should_continue,
        {
            "step7_parameter_setting": "step7_parameter_setting",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step7_parameter_setting",
        should_continue,
        {
            "step9_tool_execution": "step9_tool_execution",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step9_tool_execution",
        should_continue,
        {
            "step10_result_output": "step10_result_output",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "step10_result_output",
        should_continue,
        {
            END: END
        }
    )
    
    # 메모리 추가
    memory = MemorySaver()
    
    # 컴파일
    app = workflow.compile(checkpointer=memory)
    
    return app

# ===== 메인 실행 코드 =====

if __name__ == "__main__":
    # 에이전트 생성
    app = create_stock_agent()
    
    # 초기 상태 설정
    initial_state = {
        "user_input": "삼성전자 주식의 현재 가격을 확인하고, 최근 30일간의 가격 변동을 분석해주세요.",
        "intention_analysis": "",
        "text_plan": "",
        "parsed_plan_list": [],
        "plan_validation_result": "",
        "validation_feedback": "",
        "revised_text_plan": "",
        "step_by_step_plan": [],
        "detailed_plan": [],
        "parameter_definition_results": [],
        "tool_validation_results": [],
        "task_queue": [],
        "task_execution_results": [],
        "execution_summary": {},
        "final_answer": "",
        "error_message": "",
        "current_stage": "",
        "revision_count": 0
    }
    
    print("🚀 개선된 9단계 주식 분석 AI 에이전트 시작")
    print("=" * 60)
    
    try:
        # 에이전트 실행
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("🎉 에이전트 실행 완료!")
        print("\n📊 최종 결과:")
        print(result.get("final_answer", "결과 없음"))
        
    except Exception as e:
        print(f"\n❌ 에이전트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc() 