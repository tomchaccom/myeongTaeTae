from datetime import datetime, timedelta
import sqlite3
from stock_data_models import History



class MemoryDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("korean_stocks.db")
        self.cursor = self.conn.cursor()

        self.database = {}

        self.cursor.execute("SELECT stock_code, date, open_price, high_price, low_price, close_price, volume FROM stock_prices")

        for row in self.cursor.fetchall():
            stock_code, date, open_price, high_price, low_price, close_price, volume = row
            
            data = {
                "stock_code": stock_code,
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "close_price": close_price,
                "volume": volume,
                "date": date
            }

            if stock_code not in self.database:
                self.database[stock_code] = {date: History.from_dict(data)}
            else:
                self.database[stock_code][date] = History.from_dict(data)

        self.conn.close()
    

    def find_stock_history_by_stock_code_and_date(self, stock_code: str, date: str) -> History:
        try:
            return self.database[stock_code][date]
        except KeyError:
            return History(date=date, open_price=0, high_price=0, low_price=0, close_price=0, volume=0)

    def _date_range(self, start_date: str, end_date: str) -> list[str]:
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        while current <= end_date:
            yield current.strftime('%Y-%m-%d')
            current += timedelta(days=1)

    def find_stock_history_by_stock_code_and_date_range(self, stock_code: str, start_date: str, end_date: str) -> list[History]:
        result = []
        for date in self._date_range(start_date, end_date):
            try:
                result.append(self.database[stock_code][date])
            except KeyError:
                continue
        return result


    def find_stock_codes_by_market(self, market: str) -> list[str]:
        return list(self.database.keys())

database = MemoryDatabase()

