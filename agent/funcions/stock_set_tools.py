from typing import List
from stock_data_models import Stock


class StockSetTools:
    """주식 데이터 집합 연산 도구"""
    
    def 종목_합집합(self, 종목_리스트1: List[Stock], 종목_리스트2: List[Stock]) -> List[Stock]:
        """
        두 종목 리스트의 합집합
        
        Args:
            종목_리스트1: 첫 번째 종목 리스트
            종목_리스트2: 두 번째 종목 리스트
            
        Returns:
            List[Stock]: 합집합 종목 리스트
        """
        # 종목코드를 기준으로 중복 제거
        종목_dict = {}
        
        for stock in 종목_리스트1:
            종목_dict[stock.종목코드] = stock
        
        for stock in 종목_리스트2:
            if stock.종목코드 not in 종목_dict:
                종목_dict[stock.종목코드] = stock
        
        return list(종목_dict.values())
    
    def 종목_교집합(self, 종목_리스트1: List[Stock], 종목_리스트2: List[Stock]) -> List[Stock]:
        """
        두 종목 리스트의 교집합
        
        Args:
            종목_리스트1: 첫 번째 종목 리스트
            종목_리스트2: 두 번째 종목 리스트
            
        Returns:
            List[Stock]: 교집합 종목 리스트
        """
        코드_세트1 = {stock.종목코드 for stock in 종목_리스트1}
        코드_세트2 = {stock.종목코드 for stock in 종목_리스트2}
        
        교집합_코드 = 코드_세트1 & 코드_세트2
        
        # 첫 번째 리스트에서 교집합에 해당하는 종목들 반환
        return [stock for stock in 종목_리스트1 if stock.종목코드 in 교집합_코드]
    
    def 종목_차집합(self, 종목_리스트1: List[Stock], 종목_리스트2: List[Stock]) -> List[Stock]:
        """
        두 종목 리스트의 차집합 (리스트1 - 리스트2)
        
        Args:
            종목_리스트1: 첫 번째 종목 리스트
            종목_리스트2: 두 번째 종목 리스트
            
        Returns:
            List[Stock]: 차집합 종목 리스트
        """
        코드_세트2 = {stock.종목코드 for stock in 종목_리스트2}
        
        return [stock for stock in 종목_리스트1 if stock.종목코드 not in 코드_세트2]
    
    def 종목_XOR집합(self, 종목_리스트1: List[Stock], 종목_리스트2: List[Stock]) -> List[Stock]:
        """
        두 종목 리스트의 XOR집합 (대칭 차집합)
        
        Args:
            종목_리스트1: 첫 번째 종목 리스트
            종목_리스트2: 두 번째 종목 리스트
            
        Returns:
            List[Stock]: XOR집합 종목 리스트
        """
        코드_세트1 = {stock.종목코드 for stock in 종목_리스트1}
        코드_세트2 = {stock.종목코드 for stock in 종목_리스트2}
        
        # XOR: (A - B) ∪ (B - A)
        xor_코드 = (코드_세트1 - 코드_세트2) | (코드_세트2 - 코드_세트1)
        
        xor_종목 = []
        
        # 첫 번째 리스트에서 XOR에 해당하는 종목들
        for stock in 종목_리스트1:
            if stock.종목코드 in xor_코드:
                xor_종목.append(stock)
        
        # 두 번째 리스트에서 XOR에 해당하는 종목들 (첫 번째에 없는 것만)
        for stock in 종목_리스트2:
            if stock.종목코드 in xor_코드 and stock.종목코드 not in 코드_세트1:
                xor_종목.append(stock)
        
        return xor_종목 