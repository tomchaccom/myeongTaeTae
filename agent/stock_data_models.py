from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class History:
    """주식 거래 이력 데이터"""
    date: str  # datetime 대신 문자열로 변경
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'History':
        """딕셔너리로부터 History 객체 생성"""
        return cls(
            date=data.get('date', ''),
            open_price=float(data.get('open_price', 0)),
            high_price=float(data.get('high_price', 0)),
            low_price=float(data.get('low_price', 0)),
            close_price=float(data.get('close_price', 0)),
            volume=int(data.get('volume', 0))
        )


@dataclass 
class Stock:
    """주식 종목 데이터"""
    종목명: str
    종목코드: str
    거래이력: Dict[datetime,History]  # history -> 거래이력으로 변경
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stock':
        """딕셔너리로부터 Stock 객체 생성"""
        거래이력 = {data['date']: History.from_dict(h) for h in data['history']}
            
        return cls(
            종목명=data.get('종목명', ''),
            종목코드=data.get('종목코드', ''),
            거래이력=거래이력
        ) 