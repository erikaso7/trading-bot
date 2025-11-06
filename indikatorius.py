import ccxt
import pandas as pd
import numpy as np
import mplfinance as mpf
import requests
import datetime
import time
import os

# ================= CONFIG =================
SYMBOL = os.getenv('SYMBOL', 'XRP/USDT')
TIMEFRAMES = ['4h', '1d']
RSI_PERIOD = 14
EMA_SHORT = 5
EMA_LONG = 20
ATR_PERIOD = 14
ADX_PERIOD = 14
STOCH_RSI_PERIOD = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
CHECK_INTERVAL = 1800  # 30 min

# Telegram
TELEGRAM_TOKEN = "8592226224:AAF8cUcSTUopY40lqXucRAtkXs9HYdbi_jk"
CHAT_ID = "5628904202"

exchange = ccxt.binance({'enableRateLimit': True})

# ================= FUNCTIONS =================

def get_ohlcv(symbol, timeframe, limit=100, retries=3):
    for attempt in range(retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Bandymas {attempt+1}/{retries} nepavyko: {e}")
            time.sleep(5)
    raise Exception("Nepavyko gauti duomenų iš Binance.")


def compute_indicators(df):
    # EMA
    df['EMA5'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volume
    df['VMA20'] = df['volume'].rolling(20).mean()
    df['OBV'] = ((df['close'].diff() > 0)*df['volume'] - (df['close'].diff() < 0)*df['volume']).cumsum()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(ATR_PERIOD).mean()

    # ADX
    df['+DM'] = df['high'].diff()
    df['-DM'] = -df['low'].diff()
    df['+DM'] = df['+DM'].where((df['+DM']>0) & (df['+DM']>df['-DM']),0)
    df['-DM'] = df['-DM'].where((df['-DM']>0) & (df['-DM']>df['+DM']),0)
    df['TR_smooth'] = df['TR'].rolling(ADX_PERIOD).mean()
    df['+DI'] = 100 * (df['+DM'].rolling(ADX_PERIOD).sum() / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM'].rolling(ADX_PERIOD).sum() / df['TR_smooth'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(ADX_PERIOD).mean()

    # Bollinger Bands
    df['BB_MID'] = df['close'].rolling(BOLLINGER_WINDOW).mean()
    df['BB_STD'] = df['close'].rolling(BOLLINGER_WINDOW).std()
    df['BB_UPPER'] = df['BB_MID'] + BOLLINGER_STD * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - BOLLINGER_STD * df['BB_STD']

    # StochRSI
    df['RSI_Low'] = df['RSI'].rolling(STOCH_RSI_PERIOD).min()
    df['RSI_High'] = df['RSI'].rolling(STOCH_RSI_PERIOD).max()
    df['STOCH_RSI'] = (df['RSI'] - df['RSI_Low']) / (df['RSI_High'] - df['RSI_Low']) * 100

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df


def detect_candlestick_patterns(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pattern = None

    # Engulfing
    if prev['close'] < prev['open'] and latest['close'] > latest['open'] \
            and latest['close'] > prev['open'] and latest['open'] < prev['close']:
        pattern = 'Bulių engulfing'
    elif prev['close'] > prev['open'] and latest['close'] < latest['open'] \
            and latest['open'] > prev['close'] and latest['close'] < prev['open']:
        pattern = 'Meškų engulfing'

    # Hammer / Shooting Star / Doji
    body = abs(latest['close'] - latest['open'])
    lower_shadow = latest['open'] - latest['low'] if latest['close'] >= latest['open'] else latest['close'] - latest['low']
    upper_shadow = latest['high'] - max(latest['close'], latest['open'])
    if lower_shadow > 2 * body and upper_shadow < body:
        pattern = 'Hammer'
    if upper_shadow > 2 * body and lower_shadow < body:
        pattern = 'Shooting Star'
    if body <= (latest['high'] - latest['low']) * 0.1:
        pattern = 'Doji'

    return pattern


def plot_candles(df, pattern=None):
    df_plot = df.set_index('timestamp')
    apdict = []

    if pattern:
        latest = df_plot.iloc[-1]
        color = 'g' if 'Bulių' in pattern or 'Hammer' in pattern else 'r'
        marker = '^' if color == 'g' else 'v'
        apdict = [mpf.make_addplot(df_plot['close'].iloc[-1:], type='scatter', markersize=100, marker=marker, color=color)]

    mpf.plot(df_plot, type='candle', style='yahoo', volume=True, addplot=apdict, savefig='signal.png')
    return 'signal.png'


def analyze_market_multi(df_dict):
    scores = {"Pirkti":0, "Parduoti":0}
    latest_info = {}

    for tf, df in df_dict.items():
        latest = df.iloc[-1]
        avg_vol = df['VMA20'].iloc[-1]
        tf_score = {"Pirkti":0, "Parduoti":0}

        if latest['EMA5'] > latest['EMA20']:
            tf_score["Pirkti"] += 20
        else:
            tf_score["Parduoti"] += 20

        if latest['RSI'] < 30:
            tf_score["Pirkti"] += 20
        elif latest['RSI'] > 70:
            tf_score["Parduoti"] += 20

        if latest['MACD'] > latest['MACD_signal']:
            tf_score["Pirkti"] += 20
        else:
            tf_score["Parduoti"] += 20

        if latest['ADX'] > 25:
            tf_score["Pirkti"] += 10
            tf_score["Parduoti"] += 10

        if latest['ATR'] > df['ATR'].rolling(ATR_PERIOD).mean().iloc[-1]:
            tf_score["Pirkti"] += 5
            tf_score["Parduoti"] += 5

        if latest['volume'] > avg_vol:
            tf_score["Pirkti"] += 5
            tf_score["Parduoti"] += 5

        scores["Pirkti"] += tf_score["Pirkti"]
        scores["Parduoti"] += tf_score["Parduoti"]
        latest_info[tf] = latest

    total = scores["Pirkti"] + scores["Parduoti"]
    buy_pct = round(scores["Pirkti"] / total * 100)
    sell_pct = round(scores["Parduoti"] / total * 100)
    action = "Pirkti" if buy_pct > sell_pct else "Parduoti"

    pattern = detect_candlestick_patterns(df_dict[TIMEFRAMES[0]])
    latest = latest_info[TIMEFRAMES[0]]

    text = (
        f"📊 *{SYMBOL} ({TIMEFRAMES[0]}) atnaujinimas*\n\n"
        f"💰 Kaina: {latest['close']:.4f} USDT\n"
        f"📈 RSI: {latest['RSI']:.2f}\n"
        f"📊 Apimtis: {latest['volume']:.0f}\n"
        f"📉 Vidutinė apimtis (20): {latest['VMA20']:.0f}\n"
        f"📊 MACD: {latest['MACD']:.4f} / Signal: {latest['MACD_signal']:.4f}\n"
        f"📊 ATR: {latest['ATR']:.4f}\n"
        f"📊 ADX: {latest['ADX']:.2f}\n"
        f"📊 StochRSI: {latest['STOCH_RSI']:.2f}\n"
        f"📌 Žvakės modelis: {pattern if pattern else 'Nėra'}\n\n"
        f"📊 Tikimybė:\n"
        f" - Pirkti: {buy_pct}%\n"
        f" - Parduoti: {sell_pct}%\n\n"
        f"📌 Rekomendacija: *{action}*"
    )

    return text, pattern


def escape_markdown(text):
    escape_chars = "_*[]()~>#+-=|{}.! "
    for char in escape_chars:
        text = text.replace(char, f"\\{char}")
    return text


def send_telegram(msg, image_path=None):
    try:
        msg = escape_markdown(msg)
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as f:
                requests.post(
                    url,
                    data={"chat_id": CHAT_ID, "caption": msg, "parse_mode": "MarkdownV2"},
                    files={"photo": f}
                )
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "MarkdownV2"})
    except Exception as e:
        print(f"Telegram klaida: {e}")


# ================= MAIN LOOP =================
while True:
    try:
        print(f"[{datetime.datetime.now()}] Traukiami duomenys...")
        df_dict = {}
        for tf in TIMEFRAMES:
            df = get_ohlcv(SYMBOL, tf, limit=200)
            df = compute_indicators(df)
            df_dict[tf] = df

        msg, pattern = analyze_market_multi(df_dict)
        image_file = plot_candles(df_dict[TIMEFRAMES[0]], pattern)
        send_telegram(msg, image_file)

    except Exception as e:
        print(f"❌ Klaida: {e}")

    time.sleep(CHECK_INTERVAL)
