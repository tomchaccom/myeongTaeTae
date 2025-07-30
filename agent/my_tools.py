#!/usr/bin/env python3
"""
주식 분석 도구들
"""

import os
import sys
import json
from datetime import datetime
from langchain_core.tools import tool
import math
import yfinance as yf
import pandas as pd

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import database   
from funcions import indicator

# ===== 유틸리티 함수들 =====

def _evaluate_expression(수식: str, variables: dict[str, float]) -> float:
    """
    주어진 수식을 안전하게 평가
    
    Args:
        수식: 평가할 수식 문자열
        variables: 변수 딕셔너리
        
    Returns:
        float: 계산 결과
    """
    safe_dict = {
        '__builtins__': {},
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'pow': pow,
        **variables
    }
    # 수식 내의 한글 변수명을 안전하게 처리
    try:
        result = eval(수식, safe_dict)
        return result
    except Exception as e:
        return False

# ===== 날짜 관련 도구들 =====

@tool
def get_current_date() -> str:
    """현재 날짜와 시간 정보를 조회합니다.
    
    현재 시스템의 날짜와 시간을 YYYY-MM-DD HH:MM:SS 형식으로 반환합니다.
    
    Returns:
        str: 현재 날짜와 시간 정보 (YYYY-MM-DD HH:MM:SS 형식)
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str, variables: dict[str, float]) -> str:
    """수학 계산을 수행합니다.
    
    주어진 수학 표현식을 안전하게 계산하여 결과를 반환합니다.
    사칙연산(+, -, *, /)과 기본 수학 함수들을 지원합니다.
    
    Args:
        expression (str): 계산할 수학 표현식
            지원되는 연산자: +, -, *, /, **, %, //, (, )
            지원되는 함수:
            - abs(x): 절댓값
            - round(x, n): 반올림 (n은 소수점 자릿수, 생략 가능)
            - min(a, b, ...): 최솟값
            - max(a, b, ...): 최댓값
            - pow(x, y): 거듭제곱 (x의 y제곱)
            - sqrt(x): 제곱근
            - sin(x), cos(x), tan(x): 삼각함수 (라디안)
            - log(x): 자연로그
            - log10(x): 상용로그
            - pi: 원주율 상수
            - e: 자연상수
        variables (dict): {변수명: 변수값} 형식의 딕셔너리
    
    Returns:
        str: 계산 결과 또는 오류 메시지
        
    Examples:
        expression: 2 + 3 * 4
        variables: {}
        result: 14

        expression: (10 + 5) / 3
        variables: {}
        result: 5.0 

        expression: pow(2, 3)
        variables: {}
        result: 8

        expression: (start_price + end_price) / 100
        variables: {"start_price": 100, "end_price": 200}
        result: 3.0

        expression: close_price * 0.1
        variables: {"close_price": 100}
        result: 10.0
    """
    try:
        # 안전한 계산을 위한 허용된 함수와 상수 정의
        safe_dict = {
            '__builtins__': {},
            # 기본 연산자들은 eval에서 자동으로 처리됨
            # 수학 함수들
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'floor': math.floor,
            'ceil': math.ceil,
            # 수학 상수들
            'pi': math.pi,
            'e': math.e,
            **variables
        }
        
        # 표현식 정리 (공백 제거 등)
        expression = expression.strip()
        
        # 위험한 키워드들 차단
        dangerous_keywords = [
            'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
            '__', 'getattr', 'setattr', 'delattr', 'hasattr', 'dir', 'globals',
            'locals', 'vars', 'callable', 'compile', 'exit', 'quit'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in expression:
                return f"오류: 허용되지 않는 키워드 '{keyword}'가 포함되어 있습니다."
        
        # 계산 수행
        result = eval(expression, safe_dict)
        
        # 결과가 복소수인 경우 처리
        if isinstance(result, complex):
            if result.imag == 0:
                result = result.real
            else:
                return f"{result.real} + {result.imag}i"
        
        # 결과가 무한대나 NaN인 경우 처리
        if math.isinf(result):
            return "오류: 결과가 무한대입니다."
        elif math.isnan(result):
            return "오류: 계산 결과가 정의되지 않습니다."
        
        # 매우 큰 수는 과학적 표기법으로 표시
        if abs(result) > 1e15:
            return f"{result:.6e}"
        
        # 소수점이 있는 경우 불필요한 0 제거
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            else:
                # 소수점 15자리까지만 표시
                return f"{result:.15g}"
        
        return str(result)
        
    except ZeroDivisionError:
        return "오류: 0으로 나눌 수 없습니다."
    except ValueError as e:
        return f"오류: 잘못된 값입니다. {str(e)}"
    except TypeError as e:
        return f"오류: 잘못된 타입입니다. {str(e)}"
    except SyntaxError:
        return "오류: 수식 문법이 올바르지 않습니다."
    except NameError as e:
        return f"오류: 정의되지 않은 변수나 함수입니다. {str(e)}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def filter_stocks_by_indicator_auto(
    market: str, 
    criteria_date: str,
    indicator_start_date: str, 
    indicator_end_date: str,  
    indicator_fn: str,
    formula: str
    ) -> str:
    """지표 조건에 따라 자동으로 주식 종목을 필터링(스크리닝)합니다.

    지정된 기간의 주가 데이터로 기술적 지표를 계산하고, 기준일의 주가 정보와 함께
    주어진 수식을 평가하여 조건을 만족하는 종목만 자동으로 선별합니다.
    criteria_date 날짜의 주가 정보를 스스로 조회하여 지표 계산을 합니다.
    
    Args:
        market (str): 주식 시장 구분. "KOSPI" 또는 "KOSDAQ"
        criteria_date (str): 기준 주가 데이터 날짜. 형식: "YYYY-MM-DD" (예: "2024-01-15")
        indicator_start_date (str): 지표 계산용 시작 날짜. 형식: "YYYY-MM-DD" (예: "2024-01-01")  
        indicator_end_date (str): 지표 계산용 종료 날짜. 형식: "YYYY-MM-DD" (예: "2024-01-10")
        indicator_fn (str): 사용할 지표 함수명. 
            사용 가능한 함수:
            - "calculate_rsi": RSI 계산
            - "calculate_average_volume": 평균 거래량 계산
            - "calculate_moving_average": 이동평균 계산
            - "detect_golden_cross": 골든크로스 감지 (1.0 또는 0.0)
            - "count_golden_cross": 골든크로스 발생 횟수
            - "detect_dead_cross": 데드크로스 감지 (1.0 또는 0.0)
            - "count_dead_cross": 데드크로스 발생 횟수
            - "detect_bollinger_lower_touch": 볼린저밴드 하단 터치 (1.0 또는 0.0)
            - "detect_bollinger_upper_touch": 볼린저밴드 상단 터치 (1.0 또는 0.0)
        formula (str): 파이썬을 실행하여 평가할 수식, True or False 를 반환할 수 있도록 조건식으로 작성해야 함. 다음 변수들을 사용 가능. 이외의 변수는 절대 사용 불가:
            변수명:
            - "indicator_value": 계산된 지표 값
            - "open_price": criteria_date의 시가
            - "high_price": criteria_date의 고가  
            - "low_price": criteria_date의 저가
            - "close_price": criteria_date의 종가
            - "volume": criteria_date의 거래량
            수식 예시: "indicator_value * 0.1" < 10, "close_price / indicator_value" > 1.05, "volume + indicator_value" > 1000, "indicator_value > 30"
    
    Returns:
        str: 조건을 만족하는 종목 정보가 포함된 JSON 문자열
        
    Examples:
        # RSI가 30 이하인 KOSPI 종목 찾기 (과매도 구간)
        filter_stocks_by_indicator_auto(
            market="KOSPI",
            criteria_date="2024-01-15", 
            indicator_start_date="2024-01-01",
            indicator_end_date="2024-01-14",
            indicator_fn="calculate_rsi",
            formula="indicator_value <= 30"
        )
        
        # 종가가 20일 이동평균보다 5% 이상 높은 종목 찾기
        filter_stocks_by_indicator_auto(
            market="KOSDAQ",
            criteria_date="2024-01-15",
            indicator_start_date="2023-12-20", 
            indicator_end_date="2024-01-14",
            indicator_fn="calculate_moving_average",
            formula="(close_price / indicator_value) > 1.05"
        )
        
        # 골든크로스가 발생한 종목 찾기
        filter_stocks_by_indicator_auto(
            market="KOSPI",
            criteria_date="2024-01-15",
            indicator_start_date="2024-01-01", 
            indicator_end_date="2024-01-15",
            indicator_fn="detect_golden_cross",
            formula="indicator_value == 1.0"
        )
    """
    indicator_fn = getattr(indicator, indicator_fn)
    
    result = []
    for stock_code in database.find_stock_codes_by_market(market):
        indicator_histories = database.find_stock_history_by_stock_code_and_date_range(stock_code, indicator_start_date, indicator_end_date)
        try:
            indicator_value = indicator_fn(indicator_histories)
        except Exception as e:
            print(f"지표 계산 중 오류: {e}")
            continue
        
        criteria_history = database.find_stock_history_by_stock_code_and_date(stock_code, criteria_date)
        variables = {
            "indicator_value": indicator_value,
            "open_price": criteria_history.open_price,
            "high_price": criteria_history.high_price,
            "low_price": criteria_history.low_price,
            "close_price": criteria_history.close_price,
            "volume": criteria_history.volume
            }
        expression_result = _evaluate_expression(formula, variables)
        if expression_result:
            result.append(stock_code)
            
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@tool
def get_stock_price_history(
    stock_code: str,
    date: str,
    market: str = "KOSPI"
) -> str:
    """특정 종목의 특정 날짜 거래이력 데이터를 조회합니다.
    
    yfinance를 사용하여 한국 주식 종목의 특정 날짜 거래 데이터를 가져옵니다.
    시가, 고가, 저가, 종가, 거래량 정보를 포함합니다.
    
    Args:
        stock_code (str): 6자리 종목 코드 (예: "005930", "035720")
        date (str): 조회할 날짜. 형식: "YYYY-MM-DD" (예: "2024-01-15")
        market (str): 주식 시장 구분. "KOSPI" 또는 "KOSDAQ" (기본값: "KOSPI")
    
    Returns:
        dict: 거래이력 데이터가 포함된 JSON
        {"stock_code": "005930", "date": "2024-01-15", "open_price": 72000, "high_price": 73000, "low_price": 71500, "close_price": 72500, "volume": 1234567}
            
    Examples:
        # 삼성전자 2024년 1월 15일 거래 데이터 조회
        get_stock_price_history("005930", "2024-01-15", "KOSPI")
        
        # 카카오 2024년 1월 15일 거래 데이터 조회  
        get_stock_price_history("035720", "2024-01-15", "KOSDAQ")
    """
    try:
        # 종목 코드 검증
        if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
            raise ValueError("종목 코드는 6자리 숫자여야 합니다. (예: 005930)")
        
        # 시장 구분에 따른 접미사 추가
        if market.upper() == "KOSPI":
            ticker_symbol = f"{stock_code}.KS"
        elif market.upper() == "KOSDAQ":
            ticker_symbol = f"{stock_code}.KQ"
        else:
            return json.dumps({
                "error": "시장 구분은 'KOSPI' 또는 'KOSDAQ'이어야 합니다."
            }, ensure_ascii=False)
        
        # 날짜 형식 검증
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요. (예: 2024-01-15)")
        
        # yfinance를 사용하여 데이터 가져오기
        ticker = yf.Ticker(ticker_symbol)
        
        # 해당 날짜부터 다음날까지 데이터 요청 (1일 데이터 확보)
        hist = ticker.history(
            start=date, 
            end=(datetime.strptime(date, "%Y-%m-%d") + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        # 데이터가 있는지 확인
        if len(hist) == 0:
            raise ValueError(f"'{date}' 날짜의 거래 데이터가 없습니다.")
        
        # 첫 번째 행의 데이터 사용 (요청한 날짜의 데이터)
        row = hist.iloc[0]
        
        # 결과 데이터 구성
        result = {
            "stock_code": stock_code,
            "market": market.upper(),
            "open_price": float(row['Open']),
            "high_price": float(row['High']),
            "low_price": float(row['Low']),
            "close_price": float(row['Close']),
            "volume": int(row['Volume']),
        }
        
        return result
        
    except Exception as e:
        raise ValueError(f"데이터 조회 중 오류가 발생했습니다: {str(e)}")

