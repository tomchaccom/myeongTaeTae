#!/usr/bin/env python3
"""
JSON 파싱 함수 테스트 스크립트
"""

from tool_utils import safe_json_parse

def test_json_parsing():
    """다양한 JSON 파싱 시나리오를 테스트합니다."""
    
    test_cases = [
        # 정상적인 JSON
        {
            "name": "정상 JSON",
            "input": '{"name": "test", "value": 123}',
            "expected": {"name": "test", "value": 123}
        },
        
        # 문자열 값에 따옴표가 없는 경우
        {
            "name": "문자열 값 따옴표 누락",
            "input": '{"name": test, "value": 123}',
            "expected": {"name": "test", "value": 123}
        },
        
        # 여러 문자열 값에 따옴표가 없는 경우
        {
            "name": "여러 문자열 값 따옴표 누락",
            "input": '{"name": test, "description": sample text, "value": 123}',
            "expected": {"name": "test", "description": "sample text", "value": 123}
        },
        
        # 배열 내 문자열 값에 따옴표가 없는 경우
        {
            "name": "배열 내 문자열 따옴표 누락",
            "input": '{"items": [apple, banana, orange], "count": 3}',
            "expected": {"items": ["apple", "banana", "orange"], "count": 3}
        },
        
        # 코드 블록으로 감싸진 JSON
        {
            "name": "코드 블록 감싸진 JSON",
            "input": '```json\n{"name": test, "value": 123}\n```',
            "expected": {"name": "test", "value": 123}
        },
        
        # 설명 텍스트가 포함된 JSON
        {
            "name": "설명 텍스트 포함 JSON",
            "input": '변환 결과: {"name": test, "value": 123}',
            "expected": {"name": "test", "value": 123}
        },
        
        # boolean과 null 값이 포함된 JSON
        {
            "name": "boolean/null 값 포함",
            "input": '{"name": test, "active": true, "data": null}',
            "expected": {"name": "test", "active": True, "data": None}
        }
    ]
    
    print("🧪 JSON 파싱 함수 테스트 시작\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 {i}: {test_case['name']}")
        print(f"입력: {test_case['input']}")
        
        try:
            result = safe_json_parse(test_case['input'])
            expected = test_case['expected']
            
            if result == expected:
                print(f"✅ 성공: {result}")
            else:
                print(f"❌ 실패: 예상 {expected}, 실제 {result}")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_json_parsing() 