
from stock_data_models import History
import numpy as np



def calculate_rsi(stock_history: list[History]) -> float:
    """
    주어진 주식 히스토리에서 RSI(Relative Strength Index)를 계산합니다.

    Args:
        stock_history (List[History]): 주식 가격 데이터 리스트

    Returns:
        float: RSI 값
    """
    if len(stock_history) < 15:
        return float('nan')

    # 날짜순 정렬
    stock_history = sorted(stock_history, key=lambda x: x.date)
    closes = np.array([h.close_price for h in stock_history])
    deltas = closes[1:] - closes[:-1]

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_average_volume(history: list[History]) -> float:
    """
    주어진 주식 히스토리에서 거래량(volume)의 평균을 계산합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 거래량 평균
    """
    if not history:
        return 0.0

    volumes = [h.volume for h in history]
    avg_volume = sum(volumes) / len(volumes)
    return avg_volume


def calculate_moving_average(history: list[History]) -> float:
    """
    주어진 주식 히스토리에서 이동평균을 계산합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 이동평균 값
    """
    if not isinstance(history, list):
        history = list(history)
    closes = [h.close_price for h in history]
    avg = sum(closes) / len(closes)
    return float(avg)

SHORT_WINDOW = 5
LONG_WINDOW = 20


def detect_golden_cross(history: list[History]) -> float:
    """
    골든 크로스 여부를 감지합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 골든 크로스 발생 시 1.0, 아니면 0.0
    """
    if len(history) < LONG_WINDOW + 1:
        return 0.0

    for i in range(LONG_WINDOW, len(history)):
        # Confirm the types being passed in (should be list of History)
        prev_short_slice = history[i - SHORT_WINDOW:i]
        prev_long_slice = history[i - LONG_WINDOW:i]
        curr_short_slice = history[i - SHORT_WINDOW + 1:i + 1]
        curr_long_slice = history[i - LONG_WINDOW + 1:i + 1]

        prev_short = calculate_moving_average(prev_short_slice)      # List[History]
        prev_long = calculate_moving_average(prev_long_slice)
        curr_short = calculate_moving_average(curr_short_slice)
        curr_long = calculate_moving_average(curr_long_slice)

        # prev_short, prev_long, curr_short, curr_long는 모두 float
        if prev_short <= prev_long and curr_short > curr_long:
            return 1.0
    return 0.0



def count_golden_cross(history: list[History]) -> float:
    """
    주어진 히스토리 내에서 골든 크로스 발생 횟수를 계산합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 골든 크로스 발생 횟수
    """
    if len(history) < LONG_WINDOW + 1:
        return 0.0

    count = 0
    for i in range(LONG_WINDOW, len(history)):
        prev_short = calculate_moving_average(history[i - SHORT_WINDOW:i])
        prev_long = calculate_moving_average(history[i - LONG_WINDOW:i])
        curr_short = calculate_moving_average(history[i - SHORT_WINDOW + 1:i + 1])
        curr_long = calculate_moving_average(history[i - LONG_WINDOW + 1:i + 1])

        if prev_short <= prev_long and curr_short > curr_long:
            count += 1

    return float(count)

def detect_dead_cross(history: list[History]) -> float:
    """
    데드 크로스 여부를 감지합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 데드 크로스 발생 시 1.0, 아니면 0.0
    """
    if len(history) < LONG_WINDOW + 1:
        return 0.0

    for i in range(LONG_WINDOW, len(history)):
        prev_short = calculate_moving_average(history[i - SHORT_WINDOW:i])
        prev_long = calculate_moving_average(history[i - LONG_WINDOW:i])
        curr_short = calculate_moving_average(history[i - SHORT_WINDOW + 1:i + 1])
        curr_long = calculate_moving_average(history[i - LONG_WINDOW + 1:i + 1])

        if prev_short >= prev_long and curr_short < curr_long:
            return 1.0
    return 0.0


def count_dead_cross(history: list[History]) -> float:
    """
    주어진 히스토리 내에서 데드 크로스 발생 횟수를 계산합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 데드 크로스 발생 횟수
    """
    if len(history) < LONG_WINDOW + 1:
        return 0.0

    count = 0
    for i in range(LONG_WINDOW, len(history)):
        prev_short = calculate_moving_average(history[i - SHORT_WINDOW:i])
        prev_long = calculate_moving_average(history[i - LONG_WINDOW:i])
        curr_short = calculate_moving_average(history[i - SHORT_WINDOW + 1:i + 1])
        curr_long = calculate_moving_average(history[i - LONG_WINDOW + 1:i + 1])

        if prev_short >= prev_long and curr_short < curr_long:
            count += 1

    return float(count)


def detect_bollinger_lower_touch(history: list[History]) -> float:
    """
    볼린저 밴드 하단 터치 여부를 감지합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 하단 밴드 터치 시 1.0, 아니면 0.0
    """
    if len(history) < 20:
        return 0.0

    def safe_datetime(h):
        dt = h.date
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        elif hasattr(dt, 'iloc') or hasattr(dt, 'values'):
            if hasattr(dt, 'iloc'):
                dt = dt.iloc[0]
            else:
                dt = dt.values[0]
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
        return dt

    history_sorted = sorted(history, key=safe_datetime)

    window = history_sorted[-20:]
    ma20 = calculate_moving_average(window)
    closes = [h.close_price for h in window]
    stddev = np.std(closes)
    lower_band = ma20 - 1 * stddev
    latest_close = window[-1].close_price
    if latest_close <= lower_band:
        return 1.0
    return 0.0

def detect_bollinger_upper_touch(history: list[History]) -> float:
    """
    볼린저 밴드 상단 터치 여부를 감지합니다.

    Args:
        history (List[History]): 거래 데이터 리스트

    Returns:
        float: 상단 밴드 터치 시 1.0, 아니면 0.0
    """
    if len(history) < 20:
        return 0.0

    def safe_datetime(h):
        dt = h.date
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        elif hasattr(dt, 'iloc') or hasattr(dt, 'values'):
            if hasattr(dt, 'iloc'):
                dt = dt.iloc[0]
            else:
                dt = dt.values[0]
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
        return dt

    history_sorted = sorted(history, key=safe_datetime)

    window = history_sorted[-20:]
    ma20 = calculate_moving_average(window)
    closes = [h.close_price for h in window]
    stddev = np.std(closes)
    upper_band = ma20 + 1 * stddev
    latest_close = window[-1].close_price
    if latest_close >= upper_band:
        return 1.0
    return 0.0
