from typing import List, Dict
from stock_data_models import Stock


class StockFilterTools:
    """주식 데이터 필터링 도구"""
    
    def 단일_거래내역_종목_필터(self, 종목_리스트: List[Stock], 수식: str, 조건: str, 임계값: float) -> List[Stock]:
        """
        종목 리스트의 단일 거래내역에 대해 조건식을 만족하는 종목을 필터링
        
        Args:
            종목_리스트: 필터링할 종목 리스트
            수식: 평가할 수식 (예: "종가", "시가*1.1", "(고가+저가)/2")
            조건: 비교 조건 (">", "<", ">=", "<=", "=", "!=")
            임계값: 비교할 임계값
            
        Returns:
            List[Stock]: 필터된 종목 리스트
        """
        filtered_stocks = []
        
        for stock in 종목_리스트:
            try:
                if not stock.history:
                    continue
                
                # 최신 데이터 사용 (리스트의 마지막 항목)
                latest_data = stock.history[-1]
                
                # 수식 평가를 위한 변수 설정
                variables = {
                    '시가': latest_data.시가,
                    '고가': latest_data.고가,
                    '저가': latest_data.저가,
                    '종가': latest_data.종가,
                    '판매량': latest_data.판매량
                }
                
                # 수식 평가
                calculated_value = self._evaluate_expression(수식, variables)
                
                # 조건 평가
                if self._evaluate_condition(calculated_value, 조건, 임계값):
                    filtered_stocks.append(stock)
                    
            except Exception as e:
                print(f"종목 {stock.종목명} 필터링 중 오류: {e}")
                continue
        
        return filtered_stocks
    
    def 복수_거래내역_종목_필터(
        self, 
        비교대상_종목_리스트: List[Stock], 
        비교군_종목_리스트: List[Stock], 
        수식: str, 
        조건: str, 
        임계값: float
    ) -> List[Stock]:
        """
        두 종목 리스트를 비교하여 필터링
        
        Args:
            비교대상_종목_리스트: 필터링할 대상 종목 리스트
            비교군_종목_리스트: 비교 기준이 되는 종목 리스트
            수식: 평가할 수식
            조건: 비교 조건
            임계값: 비교할 임계값
            
        Returns:
            List[Stock]: 필터된 종목 리스트
        """
        print(수식, 조건, 임계값)

        # 비교대상 종목 리스트와 비교군 종목 리스트를 각각 딕셔너리(종목코드 기준)로 변환
        비교대상_딕트 = {stock.종목코드: stock for stock in 비교대상_종목_리스트}
        비교군_딕트 = {stock.종목코드: stock for stock in 비교군_종목_리스트}

        # 비교대상_딕트와 비교군_딕트의 key 값이 교집합인 것만 남김
        중복_코드_집합 = set(비교대상_딕트.keys()) & set(비교군_딕트.keys())

        # 종목별 비교 매핑 생성
        종목별_비교_매핑 = {}
        for stock_code in 중복_코드_집합:
            종목별_비교_매핑[stock_code] = {
                '비교군_시가': 비교군_딕트[stock_code].history[-1].시가,
                '비교군_고가': 비교군_딕트[stock_code].history[-1].고가,
                '비교군_저가': 비교군_딕트[stock_code].history[-1].저가,
                '비교군_종가': 비교군_딕트[stock_code].history[-1].종가,
                '비교군_판매량': 비교군_딕트[stock_code].history[-1].판매량,
                '비교대상_시가': 비교대상_딕트[stock_code].history[-1].시가,
                '비교대상_고가': 비교대상_딕트[stock_code].history[-1].고가,
                '비교대상_저가': 비교대상_딕트[stock_code].history[-1].저가,
                '비교대상_종가': 비교대상_딕트[stock_code].history[-1].종가,
                '비교대상_판매량': 비교대상_딕트[stock_code].history[-1].판매량
            }

        # 조건에 맞는 종목을 필터링
        filtered_stocks = []
        for stock in 비교대상_종목_리스트:
            stock_code = stock.종목코드
            if stock_code in 종목별_비교_매핑:
                try:
                    calculated_value = self._evaluate_expression(수식, 종목별_비교_매핑[stock_code])
                except ZeroDivisionError:
                    continue
                if self._evaluate_condition(calculated_value, 조건, 임계값):
                    filtered_stocks.append(stock)
        
        return filtered_stocks
    

    
    def _evaluate_condition(self, value: float, 조건: str, 임계값: float) -> bool:
        """
        조건 평가
        
        Args:
            value: 계산된 값
            조건: 비교 조건
            임계값: 비교할 임계값
            
        Returns:
            bool: 조건 만족 여부
        """
        try:
            if 조건 == '>':
                return value > 임계값
            elif 조건 == '<':
                return value < 임계값
            elif 조건 == '>=' or 조건 == '≥':
                return value >= 임계값
            elif 조건 == '<=' or 조건 == '≤':
                return value <= 임계값
            elif 조건 == '=' or 조건 == '==':
                return abs(value - 임계값) < 1e-10  # 부동소수점 비교
            elif 조건 == '!=' or 조건 == '≠':
                return abs(value - 임계값) >= 1e-10
            else:
                raise ValueError(f"지원하지 않는 조건: {조건}")
                
        except Exception as e:
            print(f"조건 평가 중 오류: {e}")
            return False 