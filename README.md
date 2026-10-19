import os
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import httpx
import pandas as pd
import ta
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from google import genai
from google.genai import types
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ================= ТВОИ ДАННЫЕ (УЖЕ ВПИСАНЫ) =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВСТАВЬ_СЮДА_КЛЮЧ_GEMINI")
TELEGRAM_BOT_TOKEN = "8840310392:AAGJFI9ds9YG_KZ-k4UWPE373j35mS7Ag_E"
TELEGRAM_CHAT_ID = "952886344"

BINANCE_FAPI_URL = "https://fapi.binance.com"
TOP_COINS_LIMIT = 40       # Топ монет по объему для проверки
MAX_AI_CALLS_PER_RUN = 3   # Сколько лучших монет разбирать за 1 цикл
MIN_CONFIDENCE = 65        # Минимальная уверенность (%) для отправки сигнала в ТГ
# =============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
SIGNALS_HISTORY = []  # Хранилище сигналов для отображения на сайте
SCAN_STATUS = {"running": False, "last_run": "Еще не запускался"}

# Твой 7-пунктовый торговый промпт
TRADING_SYSTEM_PROMPT = """Ты — мой торговый ассистент по фьючерсам и краткосрочному техническому анализу.
Твоя задача — не угадывать цену, а давать вероятностный технический разбор на основе структуры рынка и подтверждающих факторов.

Анализируй рынок по следующей логике:
- Контекст старшего таймфрейма (1H/4H): общий тренд, направление импульса, фаза рынка (тренд, коррекция, консолидация, разворот).
- Рыночная структура: HH/HL, LH/LL, слом структуры (BOS), продолжение тренда, возможный разворот.
- Ключевые уровни: поддержка, сопротивление, зоны ликвидности, зоны поглощения, имбаланс, гэпы, ретесты.
- Подтверждающие сигналы: RSI, дивергенция/конвергенция, свечные модели, импульсные свечи, ложные пробои, объемы, Funding Rate.
- Сценарий движения: bullish, bearish, нейтральный, пропуск сделки, если нет преимущества.

Верни ответ СТРОГО в формате JSON на русском языке со следующими полями:
{
  "scenario": "LONG" или "SHORT" или "ПРОПУСТИТЬ",
  "setup_quality": "сильный" или "средний" или "слабый",
  "confidence": число от 0 до 100,
  "entry": "точка или диапазон входа",
  "stop_loss": "уровень стоп-лосса",
  "tp1": "тейк-профит 1",
  "tp2": "тейк-профит 2",
  "rr": "соотношение риск/прибыль, например 1:2.5",
  "invalidation": "условия отмены сценария",
  "context": "1. Контекст рынка (тренд, откат, импульс, флэт, старший ТФ)",
  "key_zones": "2. Ключевые зоны (поддержка, сопротивление, ликвидность, имбаланс)",
  "confirmations": "3. Сигналы подтверждения (RSI, свечные формации, объемы, слом структуры)",
  "summary_reason": "7. Краткая причина решения в 1-2 предложениях"
}

Правила поведения:
- Не выдумывай сигнал, если его нет. Если рынок во флэте или рядом с серединой диапазона — выбирай ПРОПУСТИТЬ.
- Не завышай уверенность. Отдавай приоритет качеству сетапа, а не количеству сделок.
"""


async def fetch_klines(client: httpx.AsyncClient, symbol: str, interval: str, limit: int = 210) -> pd.DataFrame:
    url = f"{BINANCE_FAPI_URL}/fapi/v1/klines"
    resp = await client.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    resp.raise_for_status()
    df = pd.DataFrame(resp.json(), columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / df["vol_sma20"]
    return df


async def get_market_candidates(client: httpx.AsyncClient) -> list:
    tickers_resp = await client.get(f"{BINANCE_FAPI_URL}/fapi/v1/ticker/24hr")
    funding_resp = await client.get(f"{BINANCE_FAPI_URL}/fapi/v1/premiumIndex")

    tickers = [
        t for t in tickers_resp.json()
        if t["symbol"].endswith("USDT") and not t["symbol"].startswith("USDC")
    ]
    funding_map = {f["symbol"]: float(f.get("lastFundingRate", 0)) for f in funding_resp.json()}

    tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    top_tickers = tickers[:TOP_COINS_LIMIT]

    candidates = []
    for t in top_tickers:
        symbol = t["symbol"]
        try:
            df_15m = await fetch_klines(client, symbol, "15m", limit=210)
            last = df_15m.iloc[-1]
            rsi = float(last["rsi"])
            rvol = float(last["rvol"]) if pd.notna(last["rvol"]) else 1.0
            funding = funding_map.get(symbol, 0.0) * 100

            if rvol > 1.6 or rsi < 34 or rsi > 66:
                df_1h = await fetch_klines(client, symbol, "1h", limit=210)
                candidates.append({
                    "symbol": symbol,
                    "price": float(last["close"]),
                    "change_24h": float(t["priceChangePercent"]),
                    "funding_rate_pct": round(funding, 4),
                    "df_15m": df_15m,
                    "df_1h": df_1h,
                    "score": rvol + (abs(50 - rsi) / 15)
                })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:MAX_AI_CALLS_PER_RUN]


def build_market_summary(cand: dict) -> str:
    df_15m, df_1h = cand["df_15m"], cand["df_1h"]
    last_15m, last_1h = df_15m.iloc[-1], df_1h.iloc[-1]

    summary = {
        "symbol": cand["symbol"],
        "current_price": cand["price"],
        "24h_change_pct": cand["change_24h"],
        "funding_rate_pct": cand["funding_rate_pct"],
        "timeframe_1H": {
            "rsi_14": round(float(last_1h["rsi"]), 2),
            "ema_50": round(float(last_1h["ema50"]), 4),
            "ema_200": round(float(last_1h["ema200"]), 4),
            "atr_14": round(float(last_1h["atr"]), 4),
            "swing_high_24_bars": round(float(df_1h["high"].tail(24).max()), 4),
            "swing_low_24_bars": round(float(df_1h["low"].tail(24).min()), 4),
            "recent_12_candles": df_1h.tail(12)[["open", "high", "low", "close", "volume", "rsi"]].round(4).to_dict(orient="records")
        },
        "timeframe_15m": {
            "rsi_14": round(float(last_15m["rsi"]), 2),
            "ema_50": round(float(last_15m["ema50"]), 4),
            "ema_200": round(float(last_15m["ema200"]), 4),
            "rvol": round(float(last_15m["rvol"]), 2),
            "atr_14": round(float(last_15m["atr"]), 4),
            "swing_high_24_bars": round(float(df_15m["high"].tail(24).max()), 4),
            "swing_low_24_bars": round(float(df_15m["low"].tail(24).min()), 4),
            "recent_12_candles": df_15m.tail(12)[["open", "high", "low", "close", "volume", "rsi"]].round(4).to_dict(orient="records")
        }
    }
    return json.dumps(summary, ensure_ascii=False)


async def send_telegram_signal(client: httpx.AsyncClient, symbol: str, price: float, ai_res: dict):
    emoji = "🟢" if ai_res["scenario"] == "LONG" else "🔴"
    text = (
        f"🚨 <b>СИГНАЛ: #{symbol} (Фьючерсы)</b>\n"
        f"<b>Текущая цена:</b> <code>{price}</code>\n"
        f"<b>4. Торговый сценарий:</b> {emoji} <b>{ai_res['scenario']}</b>\n"
        f"<b>6. Оценка сетапа:</b> {ai_res['setup_quality'].upper()} (Уверенность: {ai_res['confidence']}%)\n\n"
        f"🎯 <b>5. Торговый план:</b>\n"
        f"• <b>Вход:</b> <code>{ai_res['entry']}</code>\n"
        f"• <b>Стоп-лосс:</b> <code>{ai_res['stop_loss']}</code>\n"
        f"• <b>Тейк-профит 1:</b> <code>{ai_res['tp1']}</code>\n"
        f"• <b>Тейк-профит 2:</b> <code>{ai_res['tp2']}</code>\n"
        f"• <b>RR:</b> <code>{ai_res['rr']}</code>\n"
        f"• <b>Отмена сценария:</b> {ai_res['invalidation']}\n\n"
        f"📊 <b>Технический разбор:</b>\n"
        f"<b>1. Контекст рынка:</b> {ai_res['context']}\n\n"
        f"<b>2. Ключевые зоны:</b> {ai_res['key_zones']}\n\n"
        f"<b>3. Сигналы подтверждения:</b> {ai_res['confirmations']}\n\n"
        f"📌 <b>7. Итог:</b>\n"
        f"Сценарий: {ai_res['scenario']} | Вход: {ai_res['entry']} | Стоп: {ai_res['stop_loss']} | "
        f"Тейк: {ai_res['tp1']} / {ai_res['tp2']} | RR: {ai_res['rr']} | Уверенность: {ai_res['confidence']}%\n"
        f"<b>Причина:</b> {ai_res['summary_reason']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})


async def run_screener_cycle():
    if SCAN_STATUS["running"]:
        return
    SCAN_STATUS["running"] = True
    SCAN_STATUS["last_run"] = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{SCAN_STATUS['last_run']}] 🔍 Сканирую рынок Binance Futures...")

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            candidates = await get_market_candidates(client)
            for cand in candidates:
                symbol = cand["symbol"]
                try:
                    market_json = build_market_summary(cand)
                    response = await asyncio.to_thread(
                        gemini_client.models.generate_content,
                        model="gemini-2.5-flash",
                        contents=f"Проанализируй рыночные данные фьючерсного актива и выдай JSON по 7 пунктам:\n{market_json}",
                        config=types.GenerateContentConfig(
                            system_instruction=TRADING_SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    ai_res = json.loads(response.text)
                    ai_res["symbol"] = symbol
                    ai_res["price"] = cand["price"]
                    ai_res["time"] = datetime.now().strftime("%H:%M:%S")

                    SIGNALS_HISTORY.insert(0, ai_res)
                    del SIGNALS_HISTORY[30:]  # храним последние 30 разборов

                    if ai_res.get("scenario") in ["LONG", "SHORT"] and int(ai_res.get("confidence", 0)) >= MIN_CONFIDENCE:
                        await send_telegram_signal(client, symbol, cand["price"], ai_res)
                        print(f"✅ Сигнал {symbol} ({ai_res['scenario']}) отправлен в Telegram!")
                    else:
                        print(f"⏭ {symbol}: {ai_res.get('scenario')} ({ai_res.get('confidence')}%)")
                    await asyncio.sleep(4)
                except Exception as e:
                    print(f"⚠️ Ошибка ИИ на {symbol}: {e}")
    finally:
        SCAN_STATUS["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_screener_cycle, "interval", minutes=15)
    scheduler.start()
    asyncio.create_task(run_screener_cycle())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/signals")
async def get_signals():
    return {"status": SCAN_STATUS, "signals": SIGNALS_HISTORY}


@app.post("/api/scan-now")
async def scan_now():
    if not SCAN_STATUS["running"]:
        asyncio.create_task(run_screener_cycle())
    return {"ok": True}


@app.post("/api/analyze-screenshot")
async def analyze_screenshot(symbol: str = Form("SCREENSHOT"), file: UploadFile = File(...)):
    img_bytes = await file.read()
    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=file.content_type or "image/png"),
            f"Тикер: {symbol}. Проанализируй скриншот графика строго по 7 пунктам и верни JSON."
        ],
        config=types.GenerateContentConfig(
            system_instruction=TRADING_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    ai_res = json.loads(response.text)
    ai_res["symbol"] = symbol.upper()
    ai_res["price"] = "Со скриншота"
    ai_res["time"] = datetime.now().strftime("%H:%M:%S")
    SIGNALS_HISTORY.insert(0, ai_res)

    if ai_res.get("scenario") in ["LONG", "SHORT"]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await send_telegram_signal(client, ai_res["symbol"], ai_res["price"], ai_res)
    return JSONResponse(ai_res)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
    <meta charset="UTF-8">
    <title>AI Futures Screener & Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6">
    <div class="max-w-6xl mx-auto space-y-6">
        <div class="flex flex-wrap justify-between items-center bg-slate-900 p-5 rounded-xl border border-slate-800 gap-4">
            <div>
                <h1 class="text-2xl font-bold text-white">⚡ AI Фьючерсный Скринер + ТГ Бот</h1>
                <p class="text-sm text-slate-400">Авто-сканирование Binance каждые 15 мин | Отправка в Telegram ID: 952886344</p>
            </div>
            <div class="flex items-center gap-4">
                <span id="statusText" class="text-sm text-slate-400">Загрузка...</span>
                <button onclick="triggerScan()" class="bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg font-medium text-sm transition">
                    🔍 Сканировать рынок сейчас
                </button>
            </div>
        </div>

        <!-- Ручной разбор скриншота по Ctrl+V -->
        <div class="bg-slate-900 p-5 rounded-xl border border-slate-800 space-y-3">
            <h2 class="font-semibold text-lg">📸 Разбор твоего скриншота (вставь по Ctrl+V или выбери файл)</h2>
            <div class="flex flex-wrap gap-3 items-center">
                <input id="tickerInput" type="text" placeholder="Тикер (например BTCUSDT)" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                <input id="fileInput" type="file" accept="image/*" class="text-sm text-slate-400">
                <button onclick="uploadScreenshot()" id="uploadBtn" class="bg-emerald-600 hover:bg-emerald-500 px-4 py-2 rounded-lg font-medium text-sm transition">
                    Разобрать график и прислать в ТГ
                </button>
            </div>
        </div>

        <!-- Живая лента сигналов -->
        <div id="signalsList" class="space-y-4"></div>
    </div>

    <script>
        let pastedFile = null;
        window.addEventListener('paste', e => {
            if (e.clipboardData.files.length > 0) {
                pastedFile = e.clipboardData.files[0];
                alert('Скриншот скопирован из буфера! Нажми «Разобрать график».');
            }
        });

        async function uploadScreenshot() {
            const fileInput = document.getElementById('fileInput');
            const file = pastedFile || fileInput.files[0];
            if (!file) return alert('Сначала выбери скриншот или нажми Ctrl+V!');
            const btn = document.getElementById('uploadBtn');
            btn.innerText = '⏳ ИИ анализирует...';
            const fd = new FormData();
            fd.append('file', file);
            fd.append('symbol', document.getElementById('tickerInput').value || 'CHART');
            await fetch('/api/analyze-screenshot', { method: 'POST', body: fd });
            pastedFile = null;
            btn.innerText = 'Разобрать график и прислать в ТГ';
            loadSignals();
        }

        async function triggerScan() {
            await fetch('/api/scan-now', { method: 'POST' });
            loadSignals();
        }

        async function loadSignals() {
            const res = await fetch('/api/signals');
            const data = await res.json();
            document.getElementById('statusText').innerText = data.status.running
                ? '🔄 Идет сканирование рынка...'
                : 'Последняя проверка: ' + data.status.last_run;

            const container = document.getElementById('signalsList');
            if (!data.signals.length) {
                container.innerHTML = '<div class="text-slate-500 text-center py-10">Пока нет разобранных монет. Подождите 10-20 секунд...</div>';
                return;
            }
            container.innerHTML = data.signals.map(s => {
                const badge = s.scenario === 'LONG' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                            : s.scenario === 'SHORT' ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                            : 'bg-slate-700/40 text-slate-300 border-slate-600';
                return `
                <div class="bg-slate-900 p-5 rounded-xl border border-slate-800 space-y-3">
                    <div class="flex justify-between items-center">
                        <div class="flex items-center gap-3">
                            <span class="text-xl font-bold">${s.symbol}</span>
                            <span class="text-sm text-slate-400">Цена: ${s.price}</span>
                            <span class="px-3 py-1 rounded-full text-xs font-bold border ${badge}">${s.scenario}</span>
                        </div>
                        <span class="text-xs text-slate-400">${s.time} | Сетап: <b>${s.setup_quality} (${s.confidence}%)</b></span>
                    </div>
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-2 bg-slate-950 p-3 rounded-lg text-xs">
                        <div><span class="text-slate-500">Вход:</span> <b class="text-white">${s.entry}</b></div>
                        <div><span class="text-slate-500">Стоп:</span> <b class="text-rose-400">${s.stop_loss}</b></div>
                        <div><span class="text-slate-500">Тейк 1:</span> <b class="text-emerald-400">${s.tp1}</b></div>
                        <div><span class="text-slate-500">Тейк 2:</span> <b class="text-emerald-400">${s.tp2}</b></div>
                        <div><span class="text-slate-500">RR:</span> <b class="text-indigo-400">${s.rr}</b></div>
                    </div>
                    <div class="text-sm space-y-1 text-slate-300">
                        <p><b>1. Контекст:</b> ${s.context}</p>
                        <p><b>2. Ключевые зоны:</b> ${s.key_zones}</p>
                        <p><b>3. Подтверждение:</b> ${s.confirmations}</p>
                        <p class="text-slate-400 pt-1"><b>7. Итог:</b> ${s.summary_reason}</p>
                    </div>
                </div>`;
            }).join('');
        }
        setInterval(loadSignals, 5000);
        loadSignals();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
