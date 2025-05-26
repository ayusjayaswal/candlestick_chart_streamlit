import streamlit as st
import json
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta
import os

# Set wide layout and configure page
st.set_page_config(layout="wide", page_title="Trading Analytics Platform", page_icon="📈")

# Custom CSS for better layout and chat interface
st.markdown("""
<style>
    .main > div {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .chart-title {
        color: #D1D4DC;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 8px;
        text-align: left;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 8px 8px 0px 0px;
        color: #FAFAFA;
        font-size: 16px;
        font-weight: 500;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #131722;
        color: #26a69a;
    }
    
    /* Minimal styles for suggestions */
    .stButton > button {
        width: 100%;
        text-align: left;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Define TradingView-like dark theme colors
DARK_BACKGROUND_COLOR = '#131722'
TEXT_COLOR_DARK_THEME = '#D1D4DC'
GRID_COLOR_DARK_THEME = '#363A45'

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'full_data_context' not in st.session_state:
    st.session_state.full_data_context = ""

def load_data():
    """Load and prepare data for both chart and AI agent"""
    try:
        # Load JSON for chart
        with open("data.json", "r") as f:
            json_data = json.load(f)
        
        # Load CSV for AI agent
        df = pd.read_csv("data.csv")
        
        # Create comprehensive data summary for AI context
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df.iloc[0]['timestamp'] if 'timestamp' in df.columns else "Unknown",
                "end": df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else "Unknown"
            },
            "price_stats": {
                "min": float(df['low'].min()) if 'low' in df.columns else None,
                "max": float(df['high'].max()) if 'high' in df.columns else None,
                "avg": float(df['close'].mean()) if 'close' in df.columns else None,
                "first_price": float(df['open'].iloc[0]) if 'open' in df.columns else None,
                "last_price": float(df['close'].iloc[-1]) if 'close' in df.columns else None,
            } if any(col in df.columns for col in ['high', 'low', 'close', 'open']) else None,
            "volume_stats": {
                "total": float(df['volume'].sum()) if 'volume' in df.columns else None,
                "avg": float(df['volume'].mean()) if 'volume' in df.columns else None,
                "max": float(df['volume'].max()) if 'volume' in df.columns else None,
            } if 'volume' in df.columns else None
        }
        
        # Create full data context for AI (optimized string representation)
        full_context = create_full_data_context(df, summary)
        
        st.session_state.df = df
        st.session_state.data_summary = summary
        st.session_state.full_data_context = full_context
        st.session_state.data_loaded = True
        
        return json_data, df, summary
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None

def create_full_data_context(df, summary):
    """Create optimized full dataset context for AI"""
    try:
        # Create statistical summary
        stats_context = f"""
COMPLETE DATASET ANALYSIS:
Total Records: {summary['total_rows']}
Columns: {', '.join(summary['columns'])}
Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}
"""
        
        if summary['price_stats']:
            stats_context += f"""
PRICE STATISTICS (Full Dataset):
- Minimum Price: ${summary['price_stats']['min']:.2f}
- Maximum Price: ${summary['price_stats']['max']:.2f}
- Average Price: ${summary['price_stats']['avg']:.2f}
- First Price: ${summary['price_stats']['first_price']:.2f}
- Last Price: ${summary['price_stats']['last_price']:.2f}
- Total Price Range: ${summary['price_stats']['max'] - summary['price_stats']['min']:.2f}
"""
        
        if summary['volume_stats']:
            stats_context += f"""
VOLUME STATISTICS (Full Dataset):
- Total Volume: {summary['volume_stats']['total']:,.0f}
- Average Volume: {summary['volume_stats']['avg']:,.0f}
- Maximum Volume: {summary['volume_stats']['max']:,.0f}
"""
        
        # Add key data points (first 10, last 10, and some middle points for pattern recognition)
        if len(df) > 50:
            key_data = pd.concat([
                df.head(10),           # First 10 rows
                df.iloc[len(df)//4:len(df)//4+5],  # Quarter point
                df.iloc[len(df)//2:len(df)//2+5],  # Mid point  
                df.iloc[3*len(df)//4:3*len(df)//4+5], # Three quarter point
                df.tail(10)            # Last 10 rows
            ])
        else:
            key_data = df
        
        stats_context += f"""
KEY DATA POINTS SAMPLE:
{key_data.to_string(index=False)}

FULL DATASET AVAILABLE FOR ANALYSIS - All {summary['total_rows']} records are accessible for detailed queries.
"""
        
        return stats_context
        
    except Exception as e:
        return f"Error creating data context: {str(e)}"

def setup_gemini(api_key):
    """Setup Gemini AI with optimized configuration"""
    try:
        genai.configure(api_key=api_key)
        # Use faster model configuration
        generation_config = {
            "temperature": 0.1,  # Lower temperature for more consistent responses
            "max_output_tokens": 500,  # Limit for faster responses
        }
        model = genai.GenerativeModel('gemini-2.0-flash-exp', generation_config=generation_config)
        return model
    except Exception as e:
        st.error(f"❌ Error setting up Gemini: {str(e)}")
        return None

def get_specific_data_for_query(df, question):
    """Get specific data subset based on query analysis"""
    question_lower = question.lower()
    
    # For detailed analysis requests, return more data
    if any(word in question_lower for word in ['detailed', 'full', 'complete', 'all', 'entire']):
        return df.to_string(index=False, max_rows=None)
    
    # For recent data queries
    if any(word in question_lower for word in ['recent', 'latest', 'current', 'last']):
        return df.tail(20).to_string(index=False)
    
    # For specific pattern queries
    if any(word in question_lower for word in ['highest', 'maximum', 'peak', 'top']):
        if 'high' in df.columns:
            top_data = df.nlargest(10, 'high')
            return top_data.to_string(index=False)
    
    if any(word in question_lower for word in ['lowest', 'minimum', 'bottom']):
        if 'low' in df.columns:
            bottom_data = df.nsmallest(10, 'low')
            return bottom_data.to_string(index=False)
    
    # For volume queries
    if any(word in question_lower for word in ['volume', 'trading']):
        if 'volume' in df.columns:
            volume_data = df.nlargest(15, 'volume')
            return volume_data.to_string(index=False)
    
    # Default: return recent subset
    return df.tail(15).to_string(index=False)

def query_ai_agent(model, question, df, summary, is_detailed=False):
    """Query the AI agent with full dataset access and response mode control"""
    try:
        # Determine response mode
        question_lower = question.lower()
        is_detailed_request = is_detailed or any(word in question_lower for word in 
            ['detailed', 'explain', 'why', 'how', 'analysis', 'breakdown', 'elaborate'])
        
        # Get relevant data
        specific_data = get_specific_data_for_query(df, question)
        
        # Create optimized prompt based on response mode
        if is_detailed_request:
            context_prompt = f"""
You are a professional financial analyst. You have access to a complete trading dataset.

{st.session_state.full_data_context}

SPECIFIC DATA FOR THIS QUERY:
{specific_data}

Provide a DETAILED analysis for: {question}

Guidelines:
- Give comprehensive explanations with specific numbers
- Identify trends, patterns, and correlations
- Provide actionable trading insights
- Use professional terminology
- Include supporting data points
"""
        else:
            context_prompt = f"""
You are a financial data analyst. Provide SHORT, DIRECT answers.

DATASET: {summary['total_rows']} trading records
{st.session_state.full_data_context}

RELEVANT DATA:
{specific_data}

Answer BRIEFLY and TO THE POINT: {question}

Keep response under 100 words unless specifically asked for details.
Focus on key numbers and main insights only.
"""

        response = model.generate_content(context_prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error querying AI: {str(e)}"

# Removed chat display function - no longer needed

# Create tabs
tab1, tab2 = st.tabs(["📊 Trading Chart", "🤖 AI Data Agent"])

with tab1:
    st.markdown('<div class="chart-title">TradingView Style Candlestick Chart with Bands</div>', unsafe_allow_html=True)
    
    # Load data
    json_data, df, summary = load_data()
    
    if json_data:
        from streamlit.components.v1 import html
        
        # Prepare data for chart
        candlestick_data = json_data.get("candles", [])
        markers_list_from_json = json_data.get("markers", [])
        bands_data = json_data.get("bands", [])
        
        # Process markers
        processed_markers = []
        for marker in markers_list_from_json:
            processed_marker = {
                "time": marker["time"],
                "position": marker["position"],
                "color": marker.get("color", "#2196F3"),
                "shape": marker.get("shape", "arrowUp"),
                "size": 1.0
            }
            processed_markers.append(processed_marker)
        
        # Convert to JSON strings
        candles_json_str = json.dumps(candlestick_data)
        markers_json_str = json.dumps(processed_markers)
        bands_json_str = json.dumps(bands_data) if bands_data else "[]"
        
        # Render chart
        html(f"""
        <div id="chart" style="width: 100%; height: 85vh; background: {DARK_BACKGROUND_COLOR}; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);"></div>

        <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
        <script>
          const chartElement = document.getElementById('chart');
          const darkBackgroundColor = '{DARK_BACKGROUND_COLOR}';
          const textColor = '{TEXT_COLOR_DARK_THEME}';
          const gridColor = '{GRID_COLOR_DARK_THEME}';
          
          if (!LightweightCharts) {{
            console.error("LightweightCharts library not loaded!");
          }} else {{
            const chart = LightweightCharts.createChart(chartElement, {{
                layout: {{
                    background: {{ type: 'solid', color: darkBackgroundColor }},
                    textColor: textColor,
                    fontSize: 12,
                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }},
                grid: {{ 
                    vertLines: {{ color: gridColor, style: 2, visible: true }}, 
                    horzLines: {{ color: gridColor, style: 2, visible: true }} 
                }},
                width: chartElement.clientWidth,
                height: chartElement.clientHeight,
                timeScale: {{ 
                    borderColor: gridColor, 
                    rightOffset: 50, 
                    barSpacing: 16,
                    minBarSpacing: 8,
                    timeVisible: true,
                    secondsVisible: false,
                    borderVisible: true,
                    fixLeftEdge: false,
                    fixRightEdge: false,
                }},
                priceScale: {{ 
                    borderColor: gridColor,
                    borderVisible: true,
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.1,
                    }},
                    autoScale: true,
                    entireTextOnly: false,
                }},
                crosshair: {{ 
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: {{
                        color: '#758696',
                        width: 1,
                        style: 2,
                        labelBackgroundColor: darkBackgroundColor,
                    }},
                    horzLine: {{
                        color: '#758696',
                        width: 1,
                        style: 2,
                        labelBackgroundColor: darkBackgroundColor,
                    }},
                }},
                handleScroll: true,
                handleScale: true,
            }});
            
            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: true,
                wickVisible: true,
                borderColor: '#26a69a',
                wickColor: '#737375',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                priceFormat: {{
                    type: 'price',
                    precision: 2,
                    minMove: 0.01,
                }},
            }});
            
            candlestickSeries.setData({candles_json_str});
            
            const bandsData = {bands_json_str};
            if (bandsData && bandsData.length > 0) {{
                if (bandsData.find(band => band.type === 'upper' || band.name === 'upper')) {{
                    const upperBandSeries = chart.addLineSeries({{
                        color: '#ff6b6b',
                        lineWidth: 2,
                        crosshairMarkerVisible: true,
                        priceLineVisible: false,
                        lastValueVisible: true,
                    }});
                    const upperData = bandsData.filter(band => band.type === 'upper' || band.name === 'upper')[0].data || bandsData;
                    upperBandSeries.setData(upperData);
                }}
                
                if (bandsData.find(band => band.type === 'lower' || band.name === 'lower')) {{
                    const lowerBandSeries = chart.addLineSeries({{
                        color: '#51cf66',
                        lineWidth: 2,
                        crosshairMarkerVisible: true,
                        priceLineVisible: false,
                        lastValueVisible: true,
                    }});
                    const lowerData = bandsData.filter(band => band.type === 'lower' || band.name === 'lower')[0].data || bandsData;
                    lowerBandSeries.setData(lowerData);
                }}
                
                if (bandsData.find(band => band.type === 'middle' || band.name === 'middle')) {{
                    const middleBandSeries = chart.addLineSeries({{
                        color: '#ffa726',
                        lineWidth: 2,
                        crosshairMarkerVisible: true,
                        priceLineVisible: false,                
                        lastValueVisible: true,
                    }});
                    const middleData = bandsData.filter(band => band.type === 'middle' || band.name === 'middle')[0].data || bandsData;
                    middleBandSeries.setData(middleData);
                }}
            }}
            
            let currentMarkers = {markers_json_str};
            let baseBarSpacing = 16;
            
            const updateMarkerSizes = () => {{
                const timeScale = chart.timeScale();
                const currentBarSpacing = timeScale.options().barSpacing || baseBarSpacing;
                const zoomFactor = currentBarSpacing / baseBarSpacing;
                const sizeMultiplier = Math.max(0.4, Math.min(3.0, Math.pow(zoomFactor, 0.7) * 1.2));
                
                if (currentMarkers && currentMarkers.length > 0) {{
                    const scaledMarkers = currentMarkers.map(marker => ({{
                        ...marker,
                        size: sizeMultiplier
                    }}));
                    
                    if (typeof candlestickSeries.setMarkers === 'function') {{
                        candlestickSeries.setMarkers(scaledMarkers);
                    }}
                }}
            }};
            
            if (currentMarkers && currentMarkers.length > 0 && typeof candlestickSeries.setMarkers === 'function') {{
                const cleanMarkers = currentMarkers.map(marker => ({{
                    time: marker.time,
                    position: marker.position || 'aboveBar',
                    color: marker.color || (marker.position === 'belowBar' ? '#51cf66' : '#ff6b6b'),
                    shape: marker.shape || (marker.position === 'belowBar' ? 'arrowUp' : 'arrowDown'),
                    size: 1.2
                }}));
                
                currentMarkers = cleanMarkers;
                candlestickSeries.setMarkers(cleanMarkers);
            }}
            
            chart.timeScale().subscribeVisibleTimeRangeChange(() => {{
                updateMarkerSizes();
            }});
            
            chart.timeScale().fitContent();
            
            const handleResize = () => {{
                const rect = chartElement.getBoundingClientRect();
                chart.applyOptions({{ 
                    width: rect.width, 
                    height: rect.height 
                }});
                setTimeout(updateMarkerSizes, 50);
            }};
            
            new ResizeObserver(entries => handleResize()).observe(chartElement);
            window.addEventListener('resize', handleResize);
            
            setTimeout(() => {{
                updateMarkerSizes();
                chart.timeScale().fitContent();
            }}, 200);
          }}
        </script>
        """, height=900)

with tab2:
    st.markdown("### 🤖 AI Trading Assistant")
    
    # API Key Section
    if not st.session_state.gemini_api_key:
        st.markdown("**Enter your Gemini API Key to start:**")
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("API Key", type="password", placeholder="Enter your API key...", label_visibility="collapsed")
        with col2:
            if st.button("Connect", type="primary"):
                if api_key.strip():
                    st.session_state.gemini_api_key = api_key
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
        
        st.info("💡 Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
    else:
        # API Key Status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("✅ **API Connected**")
        with col2:
            if st.button("Disconnect", type="secondary"):
                st.session_state.gemini_api_key = ""
                st.rerun()
        
        # Load data if not already loaded
        if not st.session_state.data_loaded:
            with st.spinner("Loading dataset..."):
                json_data, df, summary = load_data()
        
        if st.session_state.data_loaded:
            st.markdown("---")
            
            # Question Input
            st.markdown("**Ask about your trading data:**")
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input("Question", placeholder="e.g., What's the highest price in the dataset?", label_visibility="collapsed")
            with col2:
                send_clicked = st.button("Ask", type="primary")
            
            # Suggestions
            st.markdown("**💡 Try these questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📈 What's the current price?"):
                    user_input = "What's the current price?"
                    send_clicked = True
                if st.button("📊 Show me price trends"):
                    user_input = "Show me price trends"
                    send_clicked = True
                if st.button("💰 What's the highest price?"):
                    user_input = "What's the highest price?"
                    send_clicked = True
            with col2:
                if st.button("📉 When was the lowest price?"):
                    user_input = "When was the lowest price?"
                    send_clicked = True
                if st.button("🔥 Show highest volume days"):
                    user_input = "Show highest volume days"
                    send_clicked = True
                if st.button("📅 What's the date range?"):
                    user_input = "What's the date range of the data?"
                    send_clicked = True
            with col3:
                if st.button("📋 Summarize the data"):
                    user_input = "Summarize the trading data"
                    send_clicked = True
                if st.button("🎯 Find price patterns"):
                    user_input = "Find interesting price patterns"
                    send_clicked = True
                if st.button("⚡ Quick data overview"):
                    user_input = "Give me a quick overview"
                    send_clicked = True
            
            # Process input and show answer
            if send_clicked and user_input.strip():
                model = setup_gemini(st.session_state.gemini_api_key)
                if model:
                    st.markdown("---")
                    st.markdown(f"**Question:** {user_input}")
                    
                    with st.spinner("Analyzing data..."):
                        response = query_ai_agent(model, user_input, st.session_state.df, st.session_state.data_summary, False)
                    
                    st.markdown("**Answer:**")
                    st.write(response)
        
        else:
            st.error("❌ Could not load data. Please ensure data.csv and data.json files are available.")
