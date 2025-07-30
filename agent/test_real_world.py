#!/usr/bin/env python3
"""
실제 사용 사례를 테스트하는 스크립트
"""

from tool_utils import safe_json_parse

def test_real_world_cases():
    """실제 AI가 생성할 수 있는 JSON 형식들을 테스트합니다."""
    
    test_cases = [
        # AI가 생성한 함수 호출 JSON
        {
            "name": "AI 함수 호출 JSON",
            "input": '''
{
  "name": "filter_stocks_by_indicator_auto",
  "arguments": {
    "indicator": "PER",
    "condition": "less_than",
    "value": 15,
    "limit": 10
  }
}
''',
            "expected": {
                "name": "filter_stocks_by_indicator_auto",
                "arguments": {
                    "indicator": "PER",
                    "condition": "less_than", 
                    "value": 15,
                    "limit": 10
                }
            }
        },
        
        # 따옴표가 누락된 AI 함수 호출
        {
            "name": "따옴표 누락된 AI 함수 호출",
            "input": '''
{
  "name": filter_stocks_by_indicator_auto,
  "arguments": {
    "indicator": PER,
    "condition": less_than,
    "value": 15,
    "limit": 10
  }
}
''',
            "expected": {
                "name": "filter_stocks_by_indicator_auto",
                "arguments": {
                    "indicator": "PER",
                    "condition": "less_than",
                    "value": 15,
                    "limit": 10
                }
            }
        },
        
        # 복잡한 중첩 구조
        {
            "name": "복잡한 중첩 구조",
            "input": '''
{
  "query": {
    "type": stock_filter,
    "filters": [
      {
        "field": market_cap,
        "operator": greater_than,
        "value": 1000000000
      },
      {
        "field": sector,
        "operator": equals,
        "value": technology
      }
    ],
    "sort_by": volume,
    "order": descending
  }
}
''',
            "expected": {
                "query": {
                    "type": "stock_filter",
                    "filters": [
                        {
                            "field": "market_cap",
                            "operator": "greater_than",
                            "value": 1000000000
                        },
                        {
                            "field": "sector", 
                            "operator": "equals",
                            "value": "technology"
                        }
                    ],
                    "sort_by": "volume",
                    "order": "descending"
                }
            }
        },
        
        # 코드 블록과 설명이 포함된 실제 응답
        {
            "name": "실제 AI 응답 형식",
            "input": '''
다음은 요청하신 함수 호출입니다:

```json
{
  "name": get_stock_price_history,
  "arguments": {
    "symbol": AAPL,
    "period": 1y,
    "interval": 1d
  }
}
```

이 함수는 애플 주식의 1년간 일별 가격 데이터를 가져옵니다.
''',
            "expected": {
                "name": "get_stock_price_history",
                "arguments": {
                    "symbol": "AAPL",
                    "period": "1y",
                    "interval": "1d"
                }
            }
        }
    ]
    
    print("🌍 실제 사용 사례 테스트 시작\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 {i}: {test_case['name']}")
        print(f"입력: {test_case['input'].strip()}")
        
        try:
            result = safe_json_parse(test_case['input'])
            expected = test_case['expected']
            
            if result == expected:
                print(f"✅ 성공: {result}")
            else:
                print(f"❌ 실패: 예상 {expected}, 실제 {result}")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_real_world_cases() 