#!/usr/bin/env python3
"""
도구 관련 유틸리티 함수들
"""

import inspect
import json
import re
from typing import Any, List, Optional, Dict
from docstring_parser import parse as parse_docstring_advanced
from docstring_parser import google
from langchain.agents import tool
from pydantic import BaseModel, create_model

# 순환 import 방지를 위해 함수 내에서 import
def _get_tools():
    from my_tools import get_current_date, calculate, filter_stocks_by_indicator_auto, get_stock_price_history
    return [get_current_date, calculate, filter_stocks_by_indicator_auto, get_stock_price_history]

# ===== 도구 매핑 =====

AVAILABLE_TOOLS = _get_tools()

# 도구 이름을 키로 하는 딕셔너리
TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}


def parse_docstring(docstring):
    """함수 docstring을 전문 라이브러리로 파싱하여 설명과 파라미터 정보를 추출"""
    if not docstring:
        return {"description": "설명 없음", "parameters": {}, "returns": "반환값 설명 없음"}

    # Google docstring 스타일 명시적으로 지정
    parsed = google.parse(docstring)
    
    # 설명 추출
    description = parsed.short_description or ""
    if parsed.long_description:
        description += " " + parsed.long_description
    description = description.strip() or "설명 없음"
    
    # 파라미터 정보 추출
    parameters = {}
    for param in parsed.params:
        param_type = "string"  # 기본값
        if param.type_name:
            # 타입 이름을 JSON Schema 타입으로 변환
            type_mapping = {
                "str": "string",
                "string": "string", 
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "dict": "object",
                "list": "array"
            }
            param_type = type_mapping.get(param.type_name.lower(), param.type_name)
        
        parameters[param.arg_name] = {
            "type": param_type,
            "description": param.description or "설명 없음"
        }
    
    # 반환값 정보 추출
    returns = "반환값 설명 없음"
    if parsed.returns:
        returns = parsed.returns.description or "반환값 설명 없음"
        if parsed.returns.type_name:
            returns = f"{parsed.returns.type_name}: {returns}"
    
    return {
        "description": description,
        "parameters": parameters,
        "returns": returns
    }


def extract_signature_info(tool_func):
    """inspect 모듈을 사용해서 함수 시그니처 정보 추출"""
    signature_info = {}
    
    try:
        sig = inspect.signature(tool_func)
        
        for param_name, param in sig.parameters.items():
            param_info = {
                "name": param_name,
                "type": "string",  # 기본값
                "default": None,
                "required": True
            }
            
            # 타입 힌트 정보 추출
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if hasattr(annotation, '__name__'):
                    type_mapping = {
                        "str": "string",
                        "int": "integer", 
                        "float": "number",
                        "bool": "boolean",
                        "dict": "object",
                        "list": "array"
                    }
                    param_info["type"] = type_mapping.get(annotation.__name__, annotation.__name__)
                else:
                    param_info["type"] = str(annotation)
            
            # 기본값 정보 추출
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
                param_info["required"] = False
            
            signature_info[param_name] = param_info
            
    except Exception as e:
        print(f"⚠️ 시그니처 추출 실패: {e}")
    
    return signature_info


def get_openai_function_definitions():
    """OpenAI Function calling 표준 형식으로 도구 정보 제공"""
    function_definitions = []
    
    for tool in AVAILABLE_TOOLS:
        # 함수 객체 가져오기
        tool_func = tool.func if hasattr(tool, 'func') else None
        
        if tool_func and hasattr(tool_func, '__doc__'):
            # docstring에서 정보 추출
            parsed_doc = parse_docstring(tool_func.__doc__)
            
            # 함수 시그니처에서 추가 정보 추출
            signature_info = extract_signature_info(tool_func)
            
            # OpenAI Function calling 표준 형식
            function_def = {
                "name": tool.name,
                "description": parsed_doc["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # docstring과 시그니처 정보를 결합
            all_param_names = set(parsed_doc["parameters"].keys()) | set(signature_info.keys())
            
            for param_name in all_param_names:
                param_schema = {
                    "type": "string",
                    "description": "설명 없음"
                }
                
                # docstring 정보 우선 적용
                if param_name in parsed_doc["parameters"]:
                    doc_param = parsed_doc["parameters"][param_name]
                    param_schema["type"] = doc_param["type"]
                    param_schema["description"] = doc_param["description"]
                
                # 시그니처 정보로 보완 (타입)
                if param_name in signature_info:
                    sig_param = signature_info[param_name]
                    # docstring에 타입 정보가 없으면 시그니처에서 가져오기
                    if param_schema["type"] == "string" and sig_param["type"] != "string":
                        param_schema["type"] = sig_param["type"]
                    
                    # 기본값이 있는 경우 설명에 추가
                    if sig_param["default"] is not None:
                        param_schema["description"] += f" (기본값: {sig_param['default']})"
                
                function_def["parameters"]["properties"][param_name] = param_schema
            
            # Pydantic 스키마에서 required 필드 정보 추출
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    # Pydantic V2 방식 시도
                    schema = tool.args_schema.model_json_schema()
                except AttributeError:
                    # Pydantic V1 방식 fallback
                    schema = tool.args_schema.schema()
                
                required_fields = schema.get('required', [])
                schema_properties = schema.get('properties', {})
                
                # required 필드 목록 업데이트
                function_def["parameters"]["required"] = required_fields
                
                # 스키마에만 있는 파라미터 추가
                for param_name, param_schema in schema_properties.items():
                    if param_name not in function_def["parameters"]["properties"]:
                        function_def["parameters"]["properties"][param_name] = {
                            "type": param_schema.get('type', 'string'),
                            "description": param_schema.get('description', '설명 없음')
                        }
        else:
            # docstring이 없는 경우 기본 형식
            function_def = {
                "name": tool.name,
                "description": getattr(tool, 'description', '설명 없음'),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        
        function_definitions.append(function_def)
    
    return function_definitions


def get_tool_descriptions():
    """기존 호환성을 위한 래퍼 함수 - OpenAI 형식으로 변환"""
    openai_functions = get_openai_function_definitions()
    
    # 기존 형식으로 변환 (하위 호환성)
    tool_descriptions = []
    for func_def in openai_functions:
        tool_info = {
            "name": func_def["name"],
            "description": func_def["description"],
            "returns": "JSON 형식으로 반환",
            "parameters": {}
        }
        
        properties = func_def["parameters"].get("properties", {})
        required_fields = func_def["parameters"].get("required", [])
        
        for param_name, param_schema in properties.items():
            tool_info["parameters"][param_name] = {
                "type": param_schema["type"],
                "description": param_schema["description"],
                "required": param_name in required_fields,
                "default": None
            }
        
        tool_descriptions.append(tool_info)
    
    return tool_descriptions


def get_tools_info_string():
    """도구 정보를 문자열 형태로 포맷팅하여 반환"""
    openai_functions = get_openai_function_definitions()
    
    # OpenAI Function definitions를 JSON 형식으로 포맷팅
    tools_info_str = "다음은 OpenAI Function calling 표준 형식의 사용 가능한 도구들입니다:\n\n"
    tools_info_str += "```json\n"
    tools_info_str += json.dumps(openai_functions, ensure_ascii=False, indent=2)
    tools_info_str += "\n```\n\n"
    
    # 사용법 가이드 추가
    tools_info_str += "📋 **Function Calling 사용 규칙:**\n"
    tools_info_str += "1. 각 function의 `name`은 정확히 일치해야 합니다\n"
    tools_info_str += "2. `parameters`는 JSON Schema 형식을 따릅니다\n"
    tools_info_str += "3. `required` 배열에 있는 파라미터는 반드시 제공해야 합니다\n"
    tools_info_str += "4. 존재하지 않는 function은 사용할 수 없습니다\n"
    
    return tools_info_str


def get_tool_info_string(tool_name: str):
    """특정 도구의 정보를 문자열 형태로 포맷팅하여 반환
    
    Args:
        tool_name (str): 정보를 조회할 도구의 이름
    
    Returns:
        str: 도구 정보를 포맷팅한 문자열, 도구가 없으면 에러 메시지
    """
    openai_functions = get_openai_function_definitions()
    
    # 해당 도구 찾기
    target_function = None
    for func_def in openai_functions:
        if func_def["name"] == tool_name:
            target_function = func_def
            break
    
    if not target_function:
        return f"❌ 도구 '{tool_name}'을(를) 찾을 수 없습니다.\n\n사용 가능한 도구들: {', '.join([f['name'] for f in openai_functions])}"
    
    # 특정 도구 정보를 JSON 형식으로 포맷팅
    tools_info_str = f"'{tool_name}' 도구의 정보입니다:\n\n"
    tools_info_str += "```json\n"
    tools_info_str += json.dumps(target_function, ensure_ascii=False, indent=2)
    tools_info_str += "\n```\n\n"
    
    # 사용법 가이드 추가
    tools_info_str += "📋 **Function Calling 사용 규칙:**\n"
    tools_info_str += "1. 각 function의 `name`은 정확히 일치해야 합니다\n"
    tools_info_str += "2. `parameters`는 JSON Schema 형식을 따릅니다\n"
    tools_info_str += "3. `required` 배열에 있는 파라미터는 반드시 제공해야 합니다\n"
    tools_info_str += "4. 존재하지 않는 function은 사용할 수 없습니다\n"
    
    return tools_info_str


def safe_json_parse(content: str) -> Any:
    """
    JSON 파싱을 안전하게 수행하는 함수
    코드 블록(```json ```), 백틱, 마크다운 형식, 주석을 제거하고 JSON을 파싱합니다.
    문자열 값에 큰 따옴표가 누락된 경우도 자동으로 수정합니다.
    
    Args:
        content (str): 파싱할 문자열
        
    Returns:
        Any: 파싱된 JSON 객체
        
    Raises:
        json.JSONDecodeError: JSON 파싱에 실패한 경우
    """
    # 원본 문자열 보존
    original_content = content
    
    # 1. 문자열 양끝 공백 제거
    content = content.strip()
    
    # 2. 코드 블록 제거 (```json...``` 또는 ```...```)
    content = re.sub(r'^```(?:json)?\s*\n?(.*?)\n?```$', r'\1', content, flags=re.DOTALL)
    
    # 3. 단일 백틱 제거 (`...`)
    content = re.sub(r'^`(.*)`$', r'\1', content, flags=re.DOTALL)
    
    # 4. 주석 제거 (// 주석, /* */ 주석, # 주석)
    # // 주석 제거
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    # /* */ 주석 제거
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # # 주석 제거
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    
    # 5. LLM이 추가하는 설명 텍스트 제거
    # "변환 결과:", "처리된 결과:" 등의 텍스트 제거
    content = re.sub(r'^(변환 결과|처리된 결과|추출 결과|결과|출력|JSON|json):?\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # 6. 코드 블록 외부의 설명 텍스트 제거
    # JSON 코드 블록만 추출
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, flags=re.DOTALL)
    if json_block_match:
        content = json_block_match.group(1).strip()
    
    try:
        # 첫 번째 시도: 처리된 문자열로 파싱
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # 두 번째 시도: 문자열 값에 큰 따옴표 추가 후 파싱
            fixed_content = _fix_missing_quotes(content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            try:
                # 세 번째 시도: 더 적극적인 문자열 정리 후 파싱
                aggressively_fixed = _aggressively_fix_json(content)
                return json.loads(aggressively_fixed)
            except json.JSONDecodeError:
                try:
                    # 네 번째 시도: 원본 문자열로 파싱
                    return json.loads(original_content)
                except json.JSONDecodeError as e:
                    # 다섯 번째 시도: 더 적극적인 정리 후 파싱
                    try:
                        # 모든 줄바꿈과 공백을 정리
                        cleaned_content = re.sub(r'\s+', ' ', content)
                        # JSON 객체/배열만 추출
                        json_obj = re.search(r'(\{.*\}|\[.*\])', cleaned_content, flags=re.DOTALL)
                        if json_obj:
                            return json.loads(json_obj.group(1))
                    except:
                        pass
                    
                    # 디버깅을 위해 처리된 내용과 원본 내용을 출력
                    print(e)
                    print(f"⚠️ JSON 파싱 실패")
                    print(f"원본 내용: {original_content}")
                    print(f"처리된 내용: {content}")
                    raise e


def _fix_missing_quotes(content: str) -> str:
    """
    JSON 문자열에서 문자열 값에 큰 따옴표가 누락된 경우를 수정하는 함수
    
    Args:
        content (str): 수정할 JSON 문자열
        
    Returns:
        str: 수정된 JSON 문자열
    """
    
    def is_valid_json_value(value: str) -> bool:
        """값이 이미 유효한 JSON 값인지 확인"""
        value = value.strip()
        return (value.startswith('"') and value.endswith('"')) or \
               value.lower() in ['true', 'false', 'null'] or \
               value.replace('.', '').replace('-', '').isdigit() or \
               value.startswith('{') or value.startswith('[')
    
    # 더 강력한 정규식 패턴으로 반복 처리
    def fix_json_values(match):
        key = match.group(1)
        value = match.group(2).strip()
        
        if is_valid_json_value(value):
            return f'"{key}": {value}'
        else:
            return f'"{key}": "{value}"'
    
    # 여러 번 반복하여 모든 케이스를 처리
    for _ in range(5):  # 최대 5번 반복
        original_content = content
        
        # 1. 기본 키-값 쌍 처리
        content = re.sub(r'"([^"]+)":\s*([^,}\]]*?)(?=\s*[,}\]])', fix_json_values, content)
        
        # 2. 배열 내 값들 처리
        def fix_array_content(match):
            array_content = match.group(1)
            
            def fix_array_item(item_match):
                item = item_match.group(1).strip()
                if is_valid_json_value(item):
                    return item
                else:
                    return f'"{item}"'
            
            # 배열 내 각 항목에 따옴표 추가
            fixed_array = re.sub(r'([^,\s]+)', fix_array_item, array_content)
            return f'[{fixed_array}]'
        
        content = re.sub(r'\[([^\]]*)\]', fix_array_content, content)
        
        # 더 이상 변경사항이 없으면 종료
        if content == original_content:
            break
    
    return content


def _aggressively_fix_json(content: str) -> str:
    """
    JSON 문자열을 더 적극적으로 수정하는 함수
    모든 문자열 값에 따옴표를 추가하고, JSON 구조를 강제로 수정합니다.
    
    Args:
        content (str): 수정할 JSON 문자열
        
    Returns:
        str: 수정된 JSON 문자열
    """
    
    def is_valid_json_value(value: str) -> bool:
        """값이 이미 유효한 JSON 값인지 확인"""
        value = value.strip()
        return (value.startswith('"') and value.endswith('"')) or \
               value.lower() in ['true', 'false', 'null'] or \
               value.replace('.', '').replace('-', '').isdigit() or \
               value.startswith('{') or value.startswith('[')
    
    # 완전히 새로운 접근 방식: 문자열을 토큰으로 분해하여 재구성
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # 콜론이 있는 라인만 처리
        if ':' in line and not line.strip().startswith('//'):
            # 키-값 쌍을 찾아서 처리
            parts = line.split(':', 1)
            if len(parts) == 2:
                key_part = parts[0].strip()
                value_part = parts[1].strip().rstrip(',')
                
                # 키가 따옴표로 감싸져 있지 않으면 추가
                if not key_part.startswith('"'):
                    key_part = f'"{key_part.strip('"')}"'
                
                # 값이 따옴표로 감싸져 있지 않고, 유효한 JSON 값이 아니면 따옴표 추가
                if not is_valid_json_value(value_part):
                    value_part = f'"{value_part}"'
                
                # 콤마가 있었으면 다시 추가
                if line.strip().endswith(','):
                    fixed_line = f'  {key_part}: {value_part},'
                else:
                    fixed_line = f'  {key_part}: {value_part}'
                
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)




