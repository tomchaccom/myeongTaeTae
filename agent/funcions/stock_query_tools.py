import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Optional
from korean_stock_db import KoreanStockDB
from stock_data_models import History, Stock


class StockQueryTools:
    """주식 데이터 조회 도구"""
    
    def __init__(self, db_name: str = "korean_stocks.db"):
        """
        도구 초기화
        
        Args:
            db_name: 주식 데이터베이스 파일명
        """
        self.db_name = db_name
        self.stock_db = KoreanStockDB(db_name)
    
    def 단일_종목_거래이력_기간조회(self, 종목명: str, 시작날짜: str, 종료날짜: str) -> List[History]:
        """
        특정 종목의 주식 거래이력 기간별 조회 (yfinance 사용)
        
        Args:
            종목명: 조회할 종목명
            시작날짜: 시작 날짜 (YYYY-MM-DD)
            종료날짜: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            List[History]: 거래이력 리스트
        """
        try:
            # 한국 종목명으로 종목코드 찾기
            종목코드 = self._get_stock_code_by_name(종목명)
            if not 종목코드:
                print(f"종목명 '{종목명}'에 해당하는 종목코드를 찾을 수 없습니다.")
                return []
            
            # yfinance에서 사용할 티커 형식으로 변환
            market = self._get_market_by_code(종목코드)
            if market == 'KOSPI':
                ticker = f"{종목코드}.KS"
            elif market == 'KOSDAQ':
                ticker = f"{종목코드}.KQ"
            else:
                ticker = f"{종목코드}.KS"  # 기본값
            
            # 날짜 차이 계산하여 적절한 주기 설정
            start_date = datetime.strptime(시작날짜, '%Y-%m-%d')
            end_date = datetime.strptime(종료날짜, '%Y-%m-%d')
            date_diff = (end_date - start_date).days
            
            interval = '1d'  # 일 단위
            
            # yfinance로 데이터 조회
            stock_data = yf.Ticker(ticker)
            hist_data = stock_data.history(start=시작날짜, end=종료날짜, interval=interval)
            
            if hist_data.empty:
                print(f"종목 '{종목명}' ({ticker})의 해당 기간 데이터가 없습니다.")
                return []
            
            # History 객체 리스트로 변환
            history_list = []
            for index, row in hist_data.iterrows():
                history = History(
                    datetime=index.to_pydatetime(),
                    시가=float(row['Open']),
                    고가=float(row['High']),
                    저가=float(row['Low']),
                    종가=float(row['Close']),
                    판매량=int(row['Volume'])
                )
                history_list.append(history)
            
            return history_list
            
        except Exception as e:
            print(f"종목 데이터 조회 중 오류 발생: {e}")
            return []
    
    def 주식_데이터_조회_모든_종목(self, 시장: str, 날짜: str) -> List[Stock]:
        """
        특정 날짜의 모든 종목 데이터 조회 (stock_prices 테이블 사용)
        
        Args:
            시장: 시장 구분 ('KOSPI', 'KOSDAQ', 'ALL')
            날짜: 조회 날짜 (YYYY-MM-DD)
            
        Returns:
            List[Stock]: 종목 리스트
        """
        try:
            cursor = self.stock_db.conn.cursor()
            
            # stock_prices 테이블과 stocks 테이블을 JOIN하여 해당 날짜 데이터 조회
            if 시장.upper() == 'ALL':
                query = """
                SELECT s.code, s.name, s.market, 
                       sp.date, sp.open_price, sp.high_price, sp.low_price, sp.close_price, sp.volume
                FROM stocks s
                JOIN stock_prices sp ON s.code = sp.stock_code
                WHERE sp.date = ?
                ORDER BY s.market, s.name
                """
                cursor.execute(query, (날짜,))
            else:
                query = """
                SELECT s.code, s.name, s.market, 
                       sp.date, sp.open_price, sp.high_price, sp.low_price, sp.close_price, sp.volume
                FROM stocks s
                JOIN stock_prices sp ON s.code = sp.stock_code
                WHERE s.market = ? AND sp.date = ?
                ORDER BY s.name
                """
                cursor.execute(query, (시장.upper(), 날짜))
            
            결과_데이터 = cursor.fetchall()
            
            if not 결과_데이터:
                print(f"'{시장}' 시장의 {날짜} 날짜 데이터가 없습니다.")
                print("주가 데이터가 수집되지 않았거나 해당 날짜가 비거래일일 수 있습니다.")
                return []
            
            # 조회된 데이터를 Stock 객체로 변환
            stocks = []
            for row in 결과_데이터:
                try:
                    code, name, market, date_str, open_price, high_price, low_price, close_price, volume = row
                    
                    # 데이터 유효성 검사
                    if None in [open_price, high_price, low_price, close_price, volume]:
                        print(f"종목 {name}({code})의 {날짜} 데이터에 NULL 값이 있어 스킵합니다.")
                        continue
                    
                    # 날짜 문자열을 datetime 객체로 변환
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    history = History(
                        datetime=date_obj,
                        시가=float(open_price),
                        고가=float(high_price),
                        저가=float(low_price),
                        종가=float(close_price),
                        판매량=int(volume)
                    )
                    
                    stock = Stock(
                        종목명=name,
                        종목코드=code,
                        history=[history]
                    )
                    stocks.append(stock)
                    
                except Exception as e:
                    print(f"종목 {name}({code}) 데이터 처리 실패: {e}")
                    continue
            
            print(f"'{시장}' 시장에서 {len(stocks)}개 종목의 {날짜} 데이터 조회 완료 (데이터베이스)")
            return stocks
            
        except Exception as e:
            print(f"모든 종목 데이터 조회 중 오류 발생: {e}")
            print("stock_prices 테이블에 데이터가 없을 수 있습니다. stock_price_collector.py를 먼저 실행해주세요.")
            return []
    
    def _get_stock_code_by_name(self, 종목명: str) -> Optional[str]:
        """종목명으로 종목코드 조회"""
        try:
            cursor = self.stock_db.conn.cursor()
            query = "SELECT code FROM stocks WHERE name = ?"
            cursor.execute(query, (종목명,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"종목코드 조회 오류: {e}")
            return None
    
    def _get_market_by_code(self, 종목코드: str) -> Optional[str]:
        """종목코드로 시장 구분 조회"""
        try:
            cursor = self.stock_db.conn.cursor()
            query = "SELECT market FROM stocks WHERE code = ?"
            cursor.execute(query, (종목코드,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"시장 구분 조회 오류: {e}")
            return None
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'stock_db'):
            self.stock_db.close() 