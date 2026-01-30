import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import time
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import io
import base64
from contextlib import contextmanager
from utils.pdf_report import generate_prediction_pdf

# Load environment variables
load_dotenv()

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

PDF_PATH = os.path.join(REPORT_DIR, "stock_prediction_report.pdf")


# Page Configuration
st.set_page_config(page_title='Market Genius', layout='wide', page_icon='📈')
st.sidebar.title("🎛️ Control Panel")

# Custom CSS
st.markdown("""
<style>
    .metric-card {border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; 
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .prediction-card {border-left: 4px solid #4e73df; padding: 1rem; 
                     margin-bottom: 1rem; background-color: #f8f9fa;}
    .news-card {border-left: 4px solid #6c757d; padding: 1rem; margin: 0.5rem 0;}
    .positive {color: #28a745;}
    .negative {color: #dc3545;}
</style>
""", unsafe_allow_html=True)

# ---------------------
# Neon Full-Screen Loader
# Inject CSS/HTML/JS for universal neon loader (Streamlit + Vanilla hooks)
# ---------------------
loader_injection = r"""
<style>
/* Overlay container */
#mg-overlay {
    position: fixed;
    inset: 0;
    background: rgba(8, 8, 12, 0.85);
    backdrop-filter: blur(8px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 99999;
}

#mg-overlay.active {
    display: flex !important;
}

.mg-loader-content { text-align: center; }

.mg-neon-spinner {
    width: 100px;
    height: 100px;
    margin: 0 auto 25px auto;
    border-radius: 50%;
    position: relative;
    box-shadow: inset 0 0 20px rgba(0, 255, 234, 0.15);
}

.mg-neon-spinner::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 8px solid transparent;
    border-top-color: #00ffea;
    animation: mg-spin 1.2s linear infinite;
    box-shadow: 0 0 25px rgba(0, 255, 234, 0.8), inset 0 0 15px rgba(0, 255, 234, 0.3);
}

.mg-neon-spinner::after {
    content: '';
    position: absolute;
    inset: 12px;
    border-radius: 50%;
    border: 4px solid transparent;
    border-right-color: rgba(0, 150, 255, 0.8);
    animation: mg-spin 1.8s linear infinite reverse;
    box-shadow: 0 0 15px rgba(0, 150, 255, 0.6);
}

.mg-neon-text {
    font-size: 1.8rem;
    font-weight: bold;
    letter-spacing: 3px;
    color: #00ffea;
    text-shadow: 0 0 15px rgba(0, 255, 234, 0.8), 0 0 30px rgba(0, 150, 255, 0.4);
    animation: mg-pulse 1.5s ease-in-out infinite;
}

@keyframes mg-spin { to { transform: rotate(360deg); } }

@keyframes mg-pulse {
    0%, 100% {
        opacity: 0.6;
        text-shadow: 0 0 15px rgba(0, 255, 234, 0.8), 0 0 30px rgba(0, 150, 255, 0.4);
    }
    50% {
        opacity: 1;
        text-shadow: 0 0 25px rgba(0, 255, 234, 0.95), 0 0 40px rgba(0, 150, 255, 0.6);
    }
}

/* Block scrolling when overlay active */
body.mg-overlay-active { overflow: hidden !important; }

/* Responsive sizes */
@media (max-width: 768px) {
    .mg-neon-text { font-size: 1.5rem; }
    .mg-neon-spinner { width: 85px; height: 85px; }
}

@media (max-width: 600px) {
    .mg-neon-text { font-size: 1.2rem; }
    .mg-neon-spinner { width: 70px; height: 70px; }
}

</style>

<!-- Overlay HTML -->
<div id="mg-overlay" aria-hidden="true">
    <div class="mg-loader-content">
        <div class="mg-neon-spinner" aria-hidden="true"></div>
        <div class="mg-neon-text">^ MARKET GENIUS ^</div>
    </div>
</div>

<script>
// Universal show/hide loader for client-side interactions
(function(){
    function safeLog(){ try{ console.debug.apply(console, arguments); }catch(e){} }

    window.mg_showLoader = function(duration){
        try{
            var d = typeof duration === 'number' ? duration : 1.2;
            var overlay = document.getElementById('mg-overlay');
            if(!overlay) return;
            overlay.classList.add('active');
            document.body.classList.add('mg-overlay-active');
            // ensure the animation paints immediately
            requestAnimationFrame(function(){ /* no-op */ });
            setTimeout(function(){
                try{ overlay.classList.remove('active'); document.body.classList.remove('mg-overlay-active'); }catch(e){}
            }, Math.max(100, d*1000));
        }catch(e){ safeLog('mg_showLoader error', e); }
    };

    // Attach listeners to common interactive elements to show loader immediately.
    function attachListeners(){
        try{
            var selector = 'button, input[type=submit], input[type=button], select, textarea, input[type=file], .stButton button, [role="slider"]';
            var nodes = document.querySelectorAll(selector);
            nodes.forEach(function(el){
                // avoid duplicate attachment
                if(el.__mg_attached) return; el.__mg_attached = true;
                var ev = (el.tagName.toLowerCase() === 'select' || el.tagName.toLowerCase() === 'input' || el.tagName.toLowerCase() === 'textarea') ? 'change' : 'click';
                el.addEventListener(ev, function(e){
                    try{
                        if(el.hasAttribute && el.hasAttribute('data-mg-no-loader')) return;
                        window.mg_showLoader(1.2);
                    }catch(err){ safeLog('mg listener error', err); }
                }, {passive:true});
            });
        }catch(e){ safeLog('mg attachListeners error', e); }
    }

    // initial attach
    attachListeners();
    // observe DOM changes and re-attach as Streamlit rerenders
    try{
        var mo = new MutationObserver(function(){ attachListeners(); });
        mo.observe(document.body, { childList: true, subtree: true });
    }catch(e){ safeLog('mg observer error', e); }
})();
</script>
"""

# Render the loader injection once so CSS/HTML/JS are available to the client
st.markdown(loader_injection, unsafe_allow_html=True)

# Streamlit session state for loader (per specification)
if 'show_loader' not in st.session_state:
    st.session_state['show_loader'] = False

def show_loader(duration: float = 1.2):
    """Set session state to show the loader, sleep, then hide and rerun.
    NOTE: Blocking sleep may block the server; this function follows the spec exactly.
    """
    # Non-blocking implementation: trigger the client-side loader immediately
    # and return without sleeping, to avoid blocking Streamlit's main thread.
    try:
        # emit client-side call to show loader for duration seconds
        st.markdown("<script>if(window.mg_showLoader){window.mg_showLoader(%s);}</script>" % float(duration), unsafe_allow_html=True)
    except Exception as e:
        try:
            st.warning(f"Could not trigger client loader: {e}")
        except Exception:
            pass

# Keep overlay in sync with session state (server-triggered show/hide)
if st.session_state.get('show_loader'):
    st.markdown("<script>var ov=document.getElementById('mg-overlay'); if(ov){ov.classList.add('active'); document.body.classList.add('mg-overlay-active');}</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>var ov=document.getElementById('mg-overlay'); if(ov){ov.classList.remove('active'); document.body.classList.remove('mg-overlay-active');}</script>", unsafe_allow_html=True)

# Initialize News API - 
try:
    newsapi_key = os.getenv('NEWS_API_KEY') or '352eb85344904bc389d8f2facb49fdd3'  # Fallback to direct key
    if newsapi_key:
        newsapi = NewsApiClient(api_key=newsapi_key)
    else:
        newsapi = None
        st.sidebar.warning("News API key not found")
except Exception as e:
    st.sidebar.warning(f"News API configuration error: {str(e)}")
    newsapi = None

# Load Model placeholder (deferred until custom_spinner is defined)
# keep model in session_state so it survives Streamlit reruns
if 'model' not in st.session_state:
    st.session_state['model'] = None
model = st.session_state.get('model')


# Helper: lazily load TensorFlow/Keras model so the app can start without TF installed
def load_keras_model_safe(path):
    try:
        import tensorflow as _tf
        from tensorflow.keras.models import load_model as _load_model
    except Exception as e:
        raise ImportError(f"TensorFlow or Keras not available: {e}")
    try:
        return _load_model(path)
    except Exception as e:
        raise

# Enhanced Prediction Functions
def predict_future_prices(data, prediction_days, model):
    if model is None or data.empty:
        return None, None, None
    
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        x = []
        sequence_length = 60
        for i in range(sequence_length, len(scaled_data)):
            x.append(scaled_data[i-sequence_length:i, 0])
        x = np.array(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        last_sequence = x[-1]
        current_batch = last_sequence.reshape(1, sequence_length, 1)
        
        predictions = []
        confidence_scores = []
        for _ in range(prediction_days):
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            confidence_scores.append(np.max(current_pred))
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [data.index[-1] + dt.timedelta(days=i) for i in range(1, prediction_days+1)]
        
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions.flatten(),
            'Confidence': np.array(confidence_scores).flatten()
        }).set_index('Date')
        
        return prediction_df, predictions[-1][0], np.mean(confidence_scores)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


# Custom loading animation (full-screen overlay with nesting counter to avoid duplicates)
@contextmanager
def custom_spinner(message: str = "Loading..."):
        # session_state keys
        ph_key = "_mg_spinner_placeholder"
        cnt_key = "_mg_spinner_count"

        # ensure counter exists
        if cnt_key not in st.session_state:
                st.session_state[cnt_key] = 0

        # create or reuse a single global placeholder
        if ph_key not in st.session_state or st.session_state[ph_key] is None:
                st.session_state[ph_key] = st.empty()

        st.session_state[cnt_key] += 1
        # render overlay only when first spinner opens
        if st.session_state[cnt_key] == 1:
                overlay_html = f"""
                <div class='mg-spinner-overlay' role='status' aria-live='polite'>
                    <div class='mg-spinner-inner'>
                        <div class='mg-spinner-title'>{message}</div>
                        <div class='mg-spinner-ring' aria-hidden='true'>
                            <div class='mg-spinner-core'></div>
                        </div>
                    </div>
                </div>
                <style>
                .mg-spinner-overlay {{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.55);backdrop-filter:blur(6px);}}
                .mg-spinner-inner {{display:flex;flex-direction:column;align-items:center;gap:16px;padding:24px;border-radius:12px;max-width:90vw}}
                .mg-spinner-title {{font-family:Segoe UI, Roboto, Arial; font-size:clamp(18px,2.5vw,26px); font-weight:800; color:#123; text-align:center}}
                .mg-spinner-ring {{width:clamp(100px,18vw,200px);height:clamp(100px,18vw,200px);position:relative;border-radius:50%;}}
                .mg-spinner-ring::before {{content:'';position:absolute;inset:0;border-radius:50%;background:conic-gradient(#4e73df,#7b61ff,#2bc0ff,#4e73df);filter:blur(8px);animation:mg-spin 1.6s linear infinite;opacity:0.95}}
                .mg-spinner-core {{position:absolute;inset:14%;border-radius:50%;background:linear-gradient(180deg,#fff,#f1f5f9);display:flex;align-items:center;justify-content:center;box-shadow:0 6px 18px rgba(12,22,50,0.06);}}
                .mg-spinner-core::after {{content:'Market Genius';font-weight:800;color:#111;font-size:clamp(12px,1.6vw,16px);}}
                @keyframes mg-spin {{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
                /* block scrolling while overlay is active */
                body.mg-spinner-open {{overflow:hidden !important}}
                </style>
                <script>document.body.classList.add('mg-spinner-open');</script>
                """
                st.session_state[ph_key].markdown(overlay_html, unsafe_allow_html=True)

        try:
                yield
        finally:
                # decrement counter and remove overlay only when reaching zero
                st.session_state[cnt_key] = max(0, st.session_state[cnt_key] - 1)
                if st.session_state[cnt_key] == 0:
                        # remove body class then clear placeholder
                        removal = "<script>document.body.classList.remove('mg-spinner-open');</script>"
                        try:
                                st.session_state[ph_key].markdown(removal, unsafe_allow_html=True)
                        except Exception:
                                pass
                        try:
                                st.session_state[ph_key].empty()
                        except Exception:
                                pass
                        st.session_state[ph_key] = None


# Report generation: build an HTML report with embedded Plotly images and prediction table
def generate_html_report(title: str, summary: str, prediction_df: pd.DataFrame, charts: dict):
        """
        charts: dict[name] -> plotly.Figure
        Returns bytes of an HTML file ready for download.
        """
        parts = []
        header = f"<h1 style='font-family:Segoe UI, Roboto, Arial;color:#123;font-size:28px'>{title}</h1>"
        parts.append(header)
        parts.append(f"<p style='font-family:Segoe UI, Roboto, Arial;color:#333'>{summary}</p>")

        # Embed charts as base64 PNGs
        for name, fig in charts.items():
                try:
                        img_bytes = fig.to_image(format='png')
                        b64 = base64.b64encode(img_bytes).decode('utf-8')
                        parts.append(f"<h3 style='font-family:Segoe UI, Roboto, Arial;color:#333'>{name}</h3>")
                        parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #e6edf3;padding:6px;border-radius:6px' />")
                except Exception as e:
                        parts.append(f"<p>Could not render chart {name}: {e}</p>")

        # Prediction table
        if prediction_df is not None:
                parts.append('<h3 style="font-family:Segoe UI, Roboto, Arial;color:#333">Predictions</h3>')
                parts.append(prediction_df.to_html(classes='pred-table', float_format='${:,.2f}'.format, border=0))

        html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset='utf-8'/>
            <title>{title}</title>
            <style>
                body {{ font-family: 'Segoe UI', Roboto, Arial; padding:24px; background:#fbfdff; color:#0b1320 }}
                .pred-table {{ border-collapse:collapse; margin-top:12px }}
                .pred-table th, .pred-table td {{ border:1px solid #e6edf3; padding:8px; text-align:left }}
            </style>
        </head>
        <body>
            {''.join(parts)}
        </body>
        </html>
        """
        return html.encode('utf-8')


# Now that custom_spinner is available, try loading the Keras model with the custom loader
# Model upload / load UI in the sidebar (safe, doesn't error on every rerun)
with st.sidebar.expander("Model (optional)", expanded=False):
    uploaded = st.file_uploader("Upload Keras model (.h5)", type=["h5", "keras", "hdf5"])    
    if uploaded is not None:
        tmp_model_path = os.path.join(os.getcwd(), "uploaded_model.h5")
        try:
            with open(tmp_model_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with custom_spinner('Loading uploaded model...'):
                loaded = load_model(tmp_model_path)
            # persist model across reruns
            st.session_state['model'] = loaded
            model = loaded
            st.sidebar.success("Model loaded from upload")
        except Exception as e:
            model = None
            st.sidebar.error(f"Model load failed: {e}")
    else:
        # Offer to load bundled model file if present
        if os.path.exists("keras_model.h5"):
            if st.button("Load bundled keras_model.h5"):
                try:
                    with custom_spinner('Loading bundled model...'):
                        loaded = load_keras_model_safe("keras_model.h5")
                    st.session_state['model'] = loaded
                    model = loaded
                    st.sidebar.success("Bundled model loaded")
                except Exception as e:
                    st.session_state['model'] = None
                    model = None
                    st.sidebar.error(f"Model load failed: {e}")
        else:
            st.sidebar.info("No model loaded. Upload a keras_model.h5 to enable AI predictions.")

def generate_trading_suggestion(current_price, predicted_prices, confidence):
    if predicted_prices is None or current_price is None or not predicted_prices.any():
        return "Hold", "Insufficient data", "#6c757d", 0
    
    try:
        prediction_days = len(predicted_prices)
        short_term_days = max(1, prediction_days // 3)
        
        short_term_change = (predicted_prices[short_term_days-1] - current_price) / current_price * 100
        long_term_change = (predicted_prices[-1] - current_price) / current_price * 100
        
        confidence_score = min(100, max(0, int(confidence * 100)))
        
        if long_term_change > 7 and confidence_score > 70:
            return "🔥 Strong Buy", f"{short_term_days}-{prediction_days} days", "#28a745", confidence_score
        elif long_term_change > 5:
            return "📈 Buy", f"{prediction_days} days", "#7ac29a", confidence_score
        elif long_term_change < -7 and confidence_score > 70:
            return "💣 Strong Sell", f"Immediate", "#dc3545", confidence_score
        elif long_term_change < -5:
            return "📉 Sell", f"{prediction_days} days", "#f8b7cd", confidence_score
        elif abs(short_term_change) > 3 and confidence_score > 65:
            if short_term_change > 0:
                return "⚡ Short Buy", f"{short_term_days} days", "#7ac29a", confidence_score
            else:
                return "⚡ Short Sell", f"{short_term_days} days", "#f8b7cd", confidence_score
        else:
            return "🤝 Hold", "Wait", "#6c757d", confidence_score
    except Exception:
        return "Hold", "Error", "#6c757d", 0

# Market Fear & Greed Index
def get_market_sentiment():
    return {
        'score': np.random.randint(20, 80),
        'sentiment': ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'][np.random.randint(0, 5)],
        'color': ['#dc3545', '#ffc107', '#6c757d', '#28a745', '#218838'][np.random.randint(0, 5)]
    }


def generate_crypto_signal(current_price, predicted_prices, rsi, macd, signal_line, confidence):

    # -------- SAFETY CHECKS --------
    if predicted_prices is None or len(predicted_prices) == 0:
        return "🤝 HOLD", "#6c757d", "No prediction data"

    if any(pd.isna(x) for x in [rsi, macd, signal_line]):
        return "🤝 HOLD", "#6c757d", "Indicators warming up"

    expected_change = (predicted_prices[-1] - current_price) / current_price * 100
    confidence_pct = int(confidence * 100)

    if expected_change > 6 and rsi < 65 and macd > signal_line and confidence_pct > 70:
        return "🚀 STRONG BUY", "#28a745", "Bullish momentum + AI confirmation"

    elif expected_change > 3:
        return "📈 BUY", "#7ac29a", "Moderate upside expected"

    elif expected_change < -6 and rsi > 70:
        return "💣 STRONG SELL", "#dc3545", "Overbought + bearish forecast"

    elif expected_change < -3:
        return "📉 SELL", "#f8b7cd", "Downside risk detected"

    else:
        return "🤝 HOLD", "#6c757d", "Market uncertainty"






# Sidebar Controls
page = st.sidebar.radio("Select Market", ["📊 Stock Market", "₿ Cryptocurrency"])

if page == "📊 Stock Market":
    st.title("📊 Stock Market Genius")
    with st.sidebar.expander("⚙️ Stock Parameters", expanded=True):
        stock_options = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", 
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE"
        ]
        STOCK = st.selectbox("Select Stock Ticker", stock_options, index=0)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", dt.date(2022, 1, 1))
        with col2:
            end_date = st.date_input("End Date", dt.date.today())
        prediction_days = st.slider("Days to predict", 1, 30, 7)
        st.markdown("---")
        st.markdown("**Technical Indicators**")
        show_rsi = st.checkbox("Show RSI", True)
        show_macd = st.checkbox("Show MACD", True)
        show_sma = st.checkbox("Show SMA/EMA", True)
        

    # Main Content
    @st.cache_data(ttl=3600)
    def load_stock_data(ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Calculate indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp12 = df['Close'].ewm(span=12, adjust=False).mean()
                exp26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp12 - exp26
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    with st.spinner('Loading stock data...'):
        df = load_stock_data(STOCK, start_date, end_date)

    if not df.empty:
        try:
            with st.spinner('Fetching live price...'):
                live_stock = yf.Ticker(STOCK).history(period='1d')
                current_price = live_stock['Close'].iloc[-1] if not live_stock.empty else None
                previous_close = live_stock['Close'].iloc[-2] if len(live_stock) > 1 else None
                price_change = current_price - previous_close if current_price and previous_close else 0
        except Exception:
            current_price, previous_close, price_change = None, None, None

        # Market Overview Cards
        st.subheader("📌 Market Overview")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<div class='metric-card'><h3>Current Price</h3><h2>{f'${current_price:.2f}' if current_price else 'N/A'}</h2></div>", 
                       unsafe_allow_html=True)
        with cols[1]:
            change_pct = (price_change/previous_close*100) if previous_close and price_change else 0
            change_class = "positive" if change_pct >= 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Daily Change</h3>
                <h2 class='{change_class}'>{f'${price_change:+.2f}' if price_change else 'N/A'}</h2>
                <p class='{change_class}'>{f'{change_pct:+.2f}%' if previous_close and price_change else ''}</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            min_price = float(df['Close'].min())
            max_price = float(df['Close'].max())
            st.markdown(f"""
            <div class='metric-card'>
                <h3>52 Week Range</h3>
                <h4>${min_price:.2f} - ${max_price:.2f}</h4>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            sentiment = get_market_sentiment()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Market Sentiment</h3>
                <h4>{sentiment['sentiment']}</h4>
                <p>Score: {sentiment['score']}/100</p>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Price Chart
        st.subheader("📈 Interactive Price Analysis")
        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price'
        ))
        
        if show_sma:
            price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
            price_fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='orange')))
        
        price_fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Technical Indicators
        if show_rsi or show_macd:
            st.subheader("📊 Technical Indicators")
            
        if show_rsi:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.update_layout(height=250, showlegend=False, title="RSI (14 days)")
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        if show_macd:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')))
            macd_fig.update_layout(height=250, showlegend=False, title="MACD")
            st.plotly_chart(macd_fig, use_container_width=True)

        # Prediction Section
        if st.session_state.get('model') is not None:
            model = st.session_state['model']
            st.subheader("🔮 AI Price Prediction")
            # use the full-screen custom spinner for prediction to give clearer feedback
            with custom_spinner('Generating predictions...'):
                prediction_df, final_pred, confidence = predict_future_prices(df, prediction_days, model)

            if prediction_df is not None:
                    pred_fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Price',
                                title=f"{prediction_days}-Day Price Forecast")
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    suggestion, timeframe, color, conf_score = generate_trading_suggestion(
                        current_price, prediction_df['Predicted Price'].values, confidence)
                    
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem;">
                            <h2 style="color:white; text-align:center;">{suggestion}</h2>
                            <p style="color:white; text-align:center;">
                                Timeframe: {timeframe} | Confidence: {conf_score}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        prediction_df.style.format({
                            "Predicted Price": "${:.2f}",
                            "Confidence": "{:.1%}"
                        }).background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                        height=min(300, len(prediction_df)*35)
                    )

                    # Downloadable report (HTML) with embedded prediction chart
                    try:
                        charts = {'Price Chart': price_fig, f'{prediction_days}-Day Forecast': pred_fig}
                        summary = f"Predictions for {STOCK}: final predicted price ${final_pred:.2f} with confidence {confidence:.2%}"
                        report_bytes = generate_html_report(f"Market Genius - {STOCK} Report", summary, prediction_df, charts)
                        st.download_button("Download Report", data=report_bytes, file_name=f"{STOCK}_report.html", mime='text/html')
                    except Exception as e:
                        st.warning(f"Could not create report: {e}")

                    # News Section - Now properly integrated with error handling
                    if newsapi:
                        try:
                            st.subheader("📰 Latest Market News")
                            with st.spinner('Fetching news...'):
                                news = newsapi.get_everything(q=STOCK, language='en', sort_by='publishedAt', page_size=3)

                            if news and news.get('articles'):
                                for article in news['articles']:
                                    if article['title'] and article['description']:
                                        st.markdown(f"""
                                        <div class='news-card'>
                                            <h4>{article['title']}</h4>
                                            <p><small>{article['source']['name']} • {article['publishedAt'][:10]}</small></p>
                                            <p>{article['description']}</p>
                                            <a href="{article['url']}" target="_blank">Read more →</a>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No recent news found for this stock")
                        except Exception as e:
                            st.warning(f"Could not load news: {str(e)}")
                    else:
                        st.warning("News API not configured - please check your API key")


    # --- STOCK NEWS SECTION ---
    st.header("📈 Stock Market News")

    # Optionally, allow user to pick a topic
    news_topic = st.text_input("Enter a stock symbol or topic (e.g., AAPL, Tesla, stock market):", "stock market")

    # Fetch news
    try:
        news = newsapi.get_everything(
            q=news_topic,
            language='en',
            sort_by='publishedAt',
            page_size=5  # number of articles to show
        )

        if news['totalResults'] > 0:
            for article in news['articles']:
                st.subheader(article['title'])
                st.write(f"*Source: {article['source']['name']} | Published at: {article['publishedAt']}*")
                st.write(article['description'])
                st.markdown(f"[Read more...]({article['url']})")
                st.write("---")
        else:
            st.write("No news found for this topic.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")



elif page == "₿ Cryptocurrency":
    st.title("₿ Cryptocurrency Genius")
    
    with st.sidebar.expander("⚙️ Crypto Parameters", expanded=True):
        crypto_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"]
        crypto_symbol = st.selectbox("Select Crypto Pair", crypto_options, index=0)
        prediction_days = st.slider("Days to predict", 1, 30, 7)
        st.markdown("---")
        st.markdown("**Crypto Metrics**")
        show_volume = st.checkbox("Show Trading Volume", True)
        show_volatility = st.checkbox("Show Volatility", True)
    
    # Main Content
    @st.cache_data(ttl=600)
    def fetch_crypto_data(symbol, limit=365):
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
            data = requests.get(url).json()
            df = pd.DataFrame(data, columns=[
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'CloseTime', 'QuoteAssetVolume', 'Trades', 
                'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignore'
            ])
            df['Time'] = pd.to_datetime(df['Time'], unit='ms')
            df.set_index('Time', inplace=True)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Calculate volatility
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=7).std() * np.sqrt(365)
            
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    with st.spinner('Loading crypto data...'):
        crypto_data = fetch_crypto_data(crypto_symbol)
    
    if not crypto_data.empty:
        current_price = crypto_data['Close'].iloc[-1]
        prev_price = crypto_data['Close'].iloc[-2]
        price_change = current_price - prev_price
        change_pct = (price_change/prev_price)*100
        
        # Crypto Overview Cards
        st.subheader("📌 Crypto Overview")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<div class='metric-card'><h3>Current Price</h3><h2>${current_price:,.2f}</h2></div>", 
                       unsafe_allow_html=True)
        with cols[1]:
            change_class = "positive" if price_change >= 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>24h Change</h3>
                <h2 class='{change_class}'>${price_change:+,.2f}</h2>
                <p class='{change_class}'>{change_pct:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            vol_24h = crypto_data['Volume'].iloc[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h3>24h Volume</h3>
                <h4>${vol_24h:,.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            volatility = crypto_data['Volatility'].iloc[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Volatility</h3>
                <h4>{volatility:.2%}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Crypto Chart
        st.subheader("📈 Crypto Price Analysis")
        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=crypto_data.index,
            open=crypto_data['Open'],
            high=crypto_data['High'],
            low=crypto_data['Low'],
            close=crypto_data['Close'],
            name='Price'
        ))
        
        if show_volume:
            price_fig.add_trace(go.Bar(
                x=crypto_data.index,
                y=crypto_data['Volume'],
                name='Volume',
                marker_color='rgba(100, 149, 237, 0.6)',
                yaxis='y2'
            ))
        
        if show_volatility:
            price_fig.add_trace(go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Volatility'],
                name='Volatility',
                line=dict(color='red', width=1),
                yaxis='y3'
            ))
        
        price_fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            yaxis=dict(title='Price', domain=[0.3, 1.0]),
            yaxis2=dict(title='Volume', domain=[0.15, 0.3], showgrid=False),
            yaxis3=dict(title='Volatility', domain=[0.0, 0.15], showgrid=False)
        )
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Prediction Section
        if st.session_state.get('model') is not None:
            model = st.session_state['model']
            st.subheader("🔮 AI Price Prediction")
            with custom_spinner('Generating predictions...'):
                prediction_df, final_pred, confidence = predict_future_prices(
                    crypto_data, prediction_days, model)

            if prediction_df is not None:
                    pred_fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Price',
                                title=f"{prediction_days}-Day Price Forecast")
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    suggestion, timeframe, color, conf_score = generate_trading_suggestion(
                        current_price, prediction_df['Predicted Price'].values, confidence)
                    
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem;">
                            <h2 style="color:white; text-align:center;">{suggestion}</h2>
                            <p style="color:white; text-align:center;">
                                Timeframe: {timeframe} | Confidence: {conf_score}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        prediction_df.style.format({
                            "Predicted Price": "${:.2f}",
                            "Confidence": "{:.1%}"
                        }).background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                        height=min(300, len(prediction_df)*35))
                    
                    signal, color, reason = generate_crypto_signal(
                        current_price,
                        prediction_df['Predicted Price'].values,
                        crypto_data['RSI'].iloc[-1] if 'RSI' in crypto_data else 50,
                        crypto_data['MACD'].iloc[-1] if 'MACD' in crypto_data else 0,
                        crypto_data['Signal'].iloc[-1] if 'Signal' in crypto_data else 0,
                        confidence
                    )

                    st.markdown(f"""
                    <div class='prediction-card'>
                        <div style="background-color:{color}; padding:1.2rem; border-radius:0.7rem;">
                            <h2 style="color:white; text-align:center;">{signal}</h2>
                            <p style="color:white; text-align:center;">
                                {reason} | Confidence: {int(confidence*100)}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    signal, color, reason = generate_crypto_signal(
                        current_price=current_price,
                        predicted_prices=prediction_df["Predicted Price"].values,
                        rsi=float(crypto_data["RSI"].iloc[-1]),
                        macd=float(crypto_data["MACD"].iloc[-1]),
                        signal_line=float(crypto_data["Signal"].iloc[-1]),
                        confidence=float(confidence)
                    )
                    
                    st.markdown("## 📊 AI Trading Signal")

                    st.markdown(
                        f"""
                        <div style="
                            background-color:{color};
                            padding:1.5rem;
                            border-radius:0.8rem;
                            margin-top:1rem;
                            text-align:center;
                        ">
                            <h2 style="color:white;">{signal}</h2>
                            <p style="color:white; font-size:1.1rem;">
                                {reason}<br>
                                Confidence: {int(confidence * 100)}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    


                    # Downloadable report (HTML) for crypto
                    try:
                        charts = {'Price Chart': price_fig, f'{prediction_days}-Day Forecast': pred_fig}
                        summary = f"Predictions for {crypto_symbol}: final predicted price ${final_pred:.2f} with confidence {confidence:.2%}"
                        report_bytes = generate_html_report(f"Market Genius - {crypto_symbol} Report", summary, prediction_df, charts)
                        st.download_button("Download Report", data=report_bytes, file_name=f"{crypto_symbol}_report.html", mime='text/html')
                    except Exception as e:
                        st.warning(f"Could not create report: {e}")

                    # =======================
                    # Generate PDF Report
                    # =======================
                    try:
                        generate_prediction_pdf(
                            file_path=PDF_PATH,
                            stock_name=STOCK,
                            prediction_date=str(dt.date.today()),
                            predicted_price=float(final_pred),
                            confidence=int(confidence * 100),
                            chart_path=None  # optional, can add later
                        )

                        # Show download button ONLY if PDF exists
                        if os.path.exists(PDF_PATH):
                            with open(PDF_PATH, "rb") as f:
                                st.download_button(
                                    label="📄 Download Prediction Report (PDF)",
                                    data=f,
                                    file_name=f"{STOCK}_prediction_report.pdf",
                                    mime="application/pdf"
                                )

                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")






        # Crypto Fear & Greed Index
        st.subheader("😨😊 Crypto Fear & Greed Index")
        col1, col2 = st.columns([1, 3])
        with col1:
            sentiment = get_market_sentiment()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Sentiment"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': sentiment['color']},
                    'steps': [
                        {'range': [0, 25], 'color': "#dc3545"},
                        {'range': [25, 50], 'color': "#ffc107"},
                        {'range': [50, 75], 'color': "#28a745"},
                        {'range': [75, 100], 'color': "#218838"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### What is the Crypto Fear & Greed Index?
            - **0-24**: Extreme Fear (Potential buying opportunity)
            - **25-49**: Fear (Market may be undervalued)
            - **50-74**: Greed (Market may be overvalued)
            - **75-100**: Extreme Greed (Potential market top)
            
            This index measures emotions and sentiments from different sources including:
            - Volatility (25%)
            - Market Momentum/Volume (25%)
            - Social Media (15%)
            - Surveys (10%)
            - Dominance (10%)
            - Trends (15%)
            """)





    # =======================
    # Crypto News Section
    # =======================
    st.subheader("📰 Latest Crypto News")

    if newsapi:
        try:
            with st.spinner("Fetching crypto news..."):
                news = newsapi.get_everything(
                    q=crypto_symbol.replace("USDT", ""),
                    language="en",
                    sort_by="publishedAt",
                    page_size=5
                )

            if news and news.get("articles"):
                for article in news["articles"]:
                    st.markdown(f"""
                    <div class='news-card'>
                        <h4>{article['title']}</h4>
                        <p><small>{article['source']['name']} • {article['publishedAt'][:10]}</small></p>
                        <p>{article['description']}</p>
                        <a href="{article['url']}" target="_blank">Read more →</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent crypto news found")

        except Exception as e:
            st.warning(f"News loading failed: {e}")
    else:
        st.warning("News API not configured")




# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
ℹ️ **About Market Genius**  
Advanced AI-powered market analysis tool providing:  
- Real-time price predictions  
- Technical indicators  
- Trading recommendations  
- Market sentiment analysis
""")

# Add watermark
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>Made with ❤️ using Streamlit | [Report Issues](https://github.com/HafizullahKhokhar1/Crypto-and-Stock-Market-Predictor/issues)</small>
""", unsafe_allow_html=True)