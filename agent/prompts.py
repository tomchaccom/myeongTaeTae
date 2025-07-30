#!/usr/bin/env python3
"""
주식 분석 AI 에이전트 프롬프트 템플릿들
"""

from typing import List, Dict, Any

def get_intention_analysis_prompt(user_input: str):
    return f"""
당신은 주식 분석 전문가입니다. 사용자의 질문을 분석하여 의도를 파악해주세요.

사용자 질문: "{user_input}"

다음 형식으로 생각하는 과정을 보여주세요:

## 분석 과정 (Thinking Process)

### 1단계: 질문 해석
- 사용자가 무엇을 묻고 있는가?
- 질문의 핵심 단어들은 무엇인가?
- 명시적/암시적 요구사항은 무엇인가?

### 2단계: 컨텍스트 파악  
- 주식/금융 관련 질문인가?
- 날짜/시간 관련 정보가 필요한가?
- 데이터 조회가 필요한가, 아니면 계산/분석이 필요한가?

### 3단계: 작업 유형 분류
- 정보 조회형: 단순히 정보를 찾아서 제공
- 계산형: 데이터를 가공하거나 계산 수행  
- 분석형: 복합적인 분석이나 판단 필요
- 복합형: 여러 단계의 작업이 순차적으로 필요

## 최종 의도 분석 결과

**핵심 의도**: [한 문장으로 요약]
**핵심 키워드**: [추출된 핵심 단어들]
**작업 유형**: [정보 조회형/계산형/분석형/복합형]
**추가 고려사항**: [특별히 주의할 점이나 제약사항]"""


def get_text_planning_prompt(intention_analysis: str, user_input: str):
    return f"""
당신은 주식 분석 전문가입니다. 이전 단계에서 파악된 의도를 바탕으로 작업 계획을 생성해주세요.
생각하는 과정을 보여주세요.
도구를 활용하면 25년도 데이터까지 조회 가능합니다.

## 이전 단계 의도 분석 결과
{intention_analysis}

## 사용자 원본 질문
"{user_input}"

## 제공 도구 목록
- get_current_date: 오늘 날짜를 반환합니다.
- calculate: 수학 계산을 수행합니다. 사칙연산과 기본 수학 함수를 지원합니다.
- filter_stocks_by_indicator_auto: 주식 데이터를 필터링하여 조건에 맞는 종목을 반환합니다.
- get_stock_price_history: 특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.

## 작업 계획 수립 과정
- 위의 의도 분석에서 도출된 핵심 의도와 작업 유형을 검토하세요.
- 목표 달성을 위해 필요한 작업들을 논리적 순서로 나열하세요.
- 각 작업을 간단한 작업 단위로 분해하세요.
- 여러 도구를 사용해야 하는 경우 여러 작업으로 분해하세요. (하나의 작업은 최대 하나의 도구만 사용 가능)
- 적합한 도구가 없는 경우 더 작은 작업으로 분해 하는 것을 검토하세요.
- 제공 도구 목록에서 정확한 도구 정보 확인하세요.
- 항상 도구에 의존하지 마세요. 적절한 도구가 없다면 주어진 도구를 이용해 해결방안을 추론하세요.

## 작업 계획 리스트
위 분석을 바탕으로 해결하기 위한 작업 계획을 텍스트 리스트로 작성하세요:

1. [작업 1 설명]
2. [작업 2 설명]
3. [작업 3 설명]
...

주의: 어떤 경우에도 존재하지 않는 도구를 사용하지 마세요
"""

def get_plan_elaboration_prompt(task_number, text_plan: str):
    return f"""
당신은 주식 분석 전문가입니다. 이전 단계에서 생성된 작업 계획을 구체적인 작업 계획으로 변환해주세요.
도구를 활용하면 25년도 데이터까지 조회 가능합니다.

## 구체화할 개별 작업
작업 번호: {task_number}
작업 설명: {text_plan}

## 제공 도구 목록
- get_current_date: 오늘 날짜를 반환합니다.
- calculate: 수학 계산을 수행합니다. 사칙연산과 기본 수학 함수를 지원합니다.
- filter_stocks_by_indicator_auto: 주식 데이터를 필터링하여 조건에 맞는 종목을 반환합니다.
- get_stock_price_history: 특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.

## 작업 구체화 지침
1. 모든 작업은 반드시 위의 도구 중 하나를 사용해야 합니다.
2. 도구가 없는 작업은 적절한 도구를 선택하여 작업을 수행하세요.
3. 정보 조회, 계산, 필터링 등 모든 작업에 적합한 도구를 매핑하세요.
4. 각 작업의 목적에 맞는 가장 적절한 도구를 선택하세요.

출력은 반드시 아래 JSON 형식을 따르세요. 문자열은 반드시 큰따옴표를 사용하세요.
출력에 주석을 추가하지 마세요.
코드블럭은 사용하지 않습니다.
다른 설명 없이 결과만 출력해주세요.
값의 타입에 유의하세요. 빈 값은 빈 문자열로 표시하세요. Boolean 값은 True 또는 False로 표시하세요(파이썬 표기방식).
출력 예시의 포멧과 동일하게 출력해주세요.

출력 형식:
{{
    "task_number": 작업 번호(int),
    "task_description": 작업 설명(str),
    "tool": {{
        "name": 도구 이름(str, 하나만 가능)
    }}
}}

출력 예시:
```
{{
    "task_number": 1,
    "task_description": "2025-06-18 삼성전자 거래이력 조회",
    "tool": {{
        "name": "get_stock_price_history"
    }}
}}
```

변환 결과:"""


# ===== 4a단계: 계획 검증 프롬프트 =====
def get_plan_validation_prompt(user_input: str, detailed_plan: str):
    return f"""
당신은 주식 분석 전문가입니다. 작업 계획의 논리적 타당성과 완성도를 검증해주세요. 

## 사용자 원본 질문
"{user_input}"

## 작업 계획
{detailed_plan}

## 검증 기준
1. 논리적 순서: 작업들이 논리적으로 올바른 순서로 배열되어 있는가?
2. 완성도: 사용자 질문에 완전히 답변하기 위한 모든 단계가 포함되어 있는가?
3. 불필요한 작업: 목표와 관련 없는 불필요한 작업이 있는가?
4. 누락된 작업: 필요하지만 누락된 작업이 있는가?
5. 도구 사용: 모든 작업이 적절한 도구를 사용하고 있는가?

## 응답 형식
반드시 다음 중 하나로만 응답하세요:
- "YES: 계획이 논리적으로 타당하고 완성도가 높습니다."
- "NO: [구체적인 문제점과 개선 방향]"

검증 결과:"""

def get_result_output_prompt(user_input: str, intention: str, plan_summary: str, results_summary: str):
    return f"""
사용자 질문: {user_input}
의도 분석: {intention}
작업 계획:
{plan_summary}
실행 결과:
{results_summary}
위 결과를 바탕으로 사용자에게 친절하고 명확한 최종 답변을 작성해주세요.
JSON 데이터가 포함된 경우 핵심 정보만 추출해서 알기 쉽게 설명해주세요.
문자열은 반드시 큰따옴표를 사용하세요.
값의 타입에 유의하세요. 빈 값은 빈 문자열로 표시하세요. Boolean 값은 True 또는 False로 표시하세요(파이썬 표기방식)."""


# 에러 메시지 템플릿들
ERROR_MESSAGES = {
    "api_key_missing": "❌ CLOVASTUDIO_API_KEY 환경변수가 설정되지 않았습니다.",
    "tool_not_found": "도구를 찾을 수 없습니다.",
    "parsing_failed": "JSON 파싱에 실패했습니다.",
    "execution_failed": "작업 실행에 실패했습니다."
}


# ===== 4b단계: 도구 사용 방법 검증 프롬프트 =====
def get_tool_usage_validation_prompt(task_number, plan_info, tool_name, tool_info, parameters):
    return f"""
당신은 도구 검증 전문가입니다. 단일 작업의 도구 사용이 올바른지 검증해주세요.
도구를 활용하면 25년도 데이터까지 조회 가능합니다.

## 제공 도구 목록
{tool_info}

## 검증할 작업
작업 번호: {task_number}
작업 설명: {plan_info}
도구 이름: {tool_name}
파라미터: {parameters}

## 검증 기준
1. 도구 존재성: 사용하려는 도구가 실제로 존재하는가?
2. 파라미터 정확성: 각 도구의 필수 파라미터가 모두 제공되었는가?
3. 파라미터 유효성: 제공된 파라미터 값들이 유효한 형식과 범위인가?

## 응답 형식
반드시 다음 중 하나로만 응답하세요:
- "YES: 도구 사용이 올바른 경우"
- "NO: 올바르지 않은 경우, 구체적인 문제점과 개선 방향 제시"

생각하는 과정을 보여주세요.
"""
# ===== 4c단계: 계획 수정 프롬프트 =====  
def get_plan_revision_prompt(intention_analysis: str, text_plan: str, validation_feedback: str):
    return f"""
당신은 주식 분석 전문가입니다. 계획 검증에서 발견된 문제점을 바탕으로 텍스트 계획을 수정해주세요.
도구를 활용하면 25년도 데이터까지 조회 가능합니다.

## 이전 단계 의도 분석 결과
{intention_analysis}

## 기존 텍스트 계획
{text_plan}

## 검증 피드백
{validation_feedback}

## 제공 도구 목록
- get_current_date: 오늘 날짜를 반환합니다.
- calculate: 수학 계산을 수행합니다. 사칙연산과 기본 수학 함수를 지원합니다.
- filter_stocks_by_indicator_auto: 주식 데이터를 필터링하여 조건에 맞는 종목을 반환합니다.
- get_stock_price_history: 특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.

## 작업 계획 수립 과정
- 위의 의도 분석에서 도출된 핵심 의도와 작업 유형을 검토하세요.
- 목표 달성을 위해 필요한 작업들을 논리적 순서로 나열하세요.
- 각 작업을 간단한 작업 단위로 분해하세요.
- 여러 도구를 사용해야 하는 경우 여러 작업으로 분해하세요. (하나의 작업은 최대 하나의 도구만 사용 가능)
- 적합한 도구가 없는 경우 더 작은 작업으로 분해 하는 것을 검토하세요.
- 제공 도구 목록에서 정확한 도구 정보 확인하세요.
- 항상 도구에 의존하지 마세요. 적절한 도구가 없다면 주어진 도구를 이용해 해결방안을 추론하세요.

1. [수정된 작업 1 설명]
2. [수정된 작업 2 설명]
3. [수정된 작업 3 설명]
...

주의: 수정 사항에 대한 근거를 명확히 하세요.
"""

def get_step_by_step_plan_extraction_prompt(text_plan: str):
    """단계별 계획 추출을 위한 프롬프트"""
    return f"""
당신은 주식 분석 전문가입니다. 텍스트 계획에서 단계별 세부 계획을 추출해 리스트로 만들어주세요.
내용 손실 없이 추출해주세요.

## 텍스트 계획
{text_plan}

## 출력 형식
다음 리스트 형식으로 출력하세요. 코드블럭은 사용하지 않습니다
```
[
    "1. 작업 1 설명",
    "2. 작업 2 설명",
    "3. 작업 3 설명",
    ...
]
```

추출 결과:"""

# ===== 4d단계: 도구 수정 프롬프트 =====
def get_tool_revision_prompt(text_plan: str, task_info: str, tool_info: str, validation_feedback: str):
    return f"""
당신은 도구 사용 전문가입니다. 검증에서 발견된 도구 사용 문제점을 수정해주세요.

## 전체 작업 계획 리스트
{text_plan}

## 현재 작업 정보
{task_info}

## 도구 상세 정보
{tool_info}

## 발견된 문제점
{validation_feedback}

## 수정 지침
1. 존재하지 않는 도구를 사용하는지 확인하세요
2. 잘못된 도구를 올바른 도구로 교체하세요
3. 더 적합한 도구가 있다면 교체하세요

출력은 반드시 아래 JSON 형식을 따르세요. 문자열은 반드시 큰따옴표를 사용하세요.
또한 출력에 주석을 추가하지 마세요.
코드블럭은 사용하지 않습니다.
다른 설명 없이 결과만 출력해주세요.
값의 타입에 유의하세요. 빈 값은 빈 문자열로 표시하세요. Boolean 값은 True 또는 False로 표시하세요(파이썬 표기방식).

출력 예시:
```
{{
    "name": "get_stock_price_history",
    "parameters": {{
        "stock_code": "005930",
        "date": "2025-06-18",
        "market": "KOSPI"
    }}
}}
```

처리된 결과:"""

def get_parameter_definition_prompt(intention_analysis: str, task_number: str, task_description: str, tool_name: str, tool_info: str):
    return f"""
당신은 주식 분석 전문가입니다. 주어진 도구의 상세 정보를 보고 정확한 도구 파라미터를 정의해주세요.

## 이전 단계 의도 분석 결과
{intention_analysis}

## 현재 작업 정보
작업 번호: {task_number}
작업 설명: {task_description}  
사용할 도구: {tool_name}

## 도구 상세 정보
{tool_info}

## 파라미터 정의 지침
1. 도구 설명을 꼼꼼히 읽고 필수 파라미터와 선택적 파라미터를 구분하세요  
2. 기본값이 있는 파라미터는 더 적절한 값이 있을 때만 설정하세요
3. 현재 작업 정보를 참고하여 파라미터를 정의하세요

출력은 반드시 아래 JSON 형식을 따르세요. 문자열은 반드시 큰따옴표를 사용하세요.
출력에 주석을 추가하지 마세요.
코드블럭은 사용하지 않습니다.
다른 설명 없이 결과만 출력해주세요.
값의 타입에 유의하세요. 빈 값은 빈 문자열로 표시하세요. Boolean 값은 True 또는 False로 표시하세요(파이썬 표기방식).


출력 예시:
```
{{
    "parameters": {{
        "stock_code": "005930",
        "date": "2025-06-18",
        "market": "KOSPI"
    }}
}}
```

처리된 결과:"""

# 상태 메시지 템플릿들
STATUS_MESSAGES = {
    "step1_start": "🧠 1단계: 의도 파악 중...",
    "step2_start": "📝 2단계: 텍스트 계획 생성 중...",
    "step2b_start": "🔍 2b단계: 단계별 계획 추출 중...",
    "step3_start": "📋 3단계: 계획 구체화 중...",
    "step3b_start": "⚙️ 3b단계: 도구 파라미터 정의 중...",
    "step4a_start": "🔍 4a단계: 계획 검증 중...",
    "step4b_start": "🛠️ 4b단계: 도구 사용 방법 검증 중...",
    "step4c_start": "📝 4c단계: 계획 수정 중...",
    "step4d_start": "🔧 4d단계: 도구 수정 중...",
    "step5_start": "⚙️ 5단계: 작업 수행 중...",
    "step6_start": "📄 6단계: 결과 정리 중...",
    "task_success": "✅ 작업 완료",
    "task_failure": "❌ 작업 실패",
    "plan_approved": "✅ 계획 승인됨",
    "plan_validation_passed": "✅ 계획 검증 통과",
    "tool_validation_passed": "✅ 도구 검증 통과", 
    "plan_validation_failed": "❌ 계획 검증 실패",
    "tool_validation_failed": "❌ 도구 검증 실패",
    "plan_needs_revision": "⚠️ 계획 수정 필요",
    "plan_needs_recreation": "❌ 계획 재작성 필요"
} 