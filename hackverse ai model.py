import os
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# 1. WINDOWS CONFIGURATION & PATH SETUP
# ---------------------------------------------------------------------------

# Local folder path using Windows raw string format (r"...")
DATA_DIR = "./extracted_data/NifSent/NIFTY 50"

# Set your Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# ---------------------------------------------------------------------------
# 2. FILE TRAVERSAL HELPERS
# ---------------------------------------------------------------------------

def load_technical_data(ticker_symbol: str, base_dir: str = ".") -> pd.DataFrame:
    """Recursively searches for the ticker CSV inside the workspace."""
    target_filename = f"{ticker_symbol.upper()}.csv"
    file_path = None

    for root, dirs, files in os.walk(base_dir):
        if target_filename in files:
            file_path = os.path.join(root, target_filename)
            break

    if not file_path:
        raise FileNotFoundError(f"Could not find {target_filename} anywhere inside {base_dir}")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def load_sentiment_data(ticker_symbol: str, base_dir: str = ".") -> pd.DataFrame:
    """Recursively searches for final_news_sentiment_analysis.csv inside the workspace."""
    target_filename = "final_news_sentiment_analysis.csv"
    file_path = None

    for root, dirs, files in os.walk(base_dir):
        if target_filename in files:
            file_path = os.path.join(root, target_filename)
            break

    if not file_path:
        raise FileNotFoundError(f"Could not find {target_filename} anywhere inside {base_dir}")

    df = pd.read_csv(file_path)

    # Filter by stock column (checking 'Stock', 'Ticker', or 'Symbol')
    if 'Stock' in df.columns:
        df = df[df['Stock'].str.upper() == ticker_symbol.upper()].reset_index(drop=True)
    elif 'Ticker' in df.columns:
        df = df[df['Ticker'].str.upper() == ticker_symbol.upper()].reset_index(drop=True)
    elif 'Symbol' in df.columns:
        df = df[df['Symbol'].str.upper() == ticker_symbol.upper()].reset_index(drop=True)

    return df
# ---------------------------------------------------------------------------
# 3. TECHNICAL AI AGENT
# ---------------------------------------------------------------------------

class TechnicalAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def analyze(self, ticker: str, df: pd.DataFrame) -> str:
        processed_df = self.compute_indicators(df)

        # Slices only 5 rows and essential columns as a tiny text string
        summary_df = processed_df[['Close', 'SMA_20', 'RSI']].tail(5)
        data_str = summary_df.to_string()

        prompt = PromptTemplate.from_template("""
        You are an expert Technical Analysis AI Agent.
        Analyze the following recent 5-day market summary for {ticker}:

        {market_data}

        Provide a structured signal output strictly in valid JSON format with keys:
        {{
            "technical_signal": "BULLISH" | "BEARISH" | "NEUTRAL",
            "confidence": 0.8,
            "reasoning": "<short justification>"
        }}
        """)

        chain = prompt | self.llm
        return chain.invoke({"ticker": ticker, "market_data": data_str}).content

# ---------------------------------------------------------------------------
# 4. SENTIMENT AI AGENT
# ---------------------------------------------------------------------------

class SentimentAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, ticker: str, news_df: pd.DataFrame) -> str:
        if news_df.empty:
            headlines = ["No recent news headlines available."]
        else:
            headlines = news_df['Headline'].tail(5).tolist()

        prompt = PromptTemplate.from_template("""
        You are a Sentiment Analysis AI Agent specializing in Indian equity markets.
        Evaluate the potential market impact of these recent headlines for {ticker}:
        {headlines}

        Provide a structured signal output strictly in valid JSON format with keys:
        {{
            "sentiment_signal": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "confidence": <float between 0.0 and 1.0>,
            "reasoning": "<1-2 sentence explanation summarizing market drivers>"
        }}
        """)

        chain = prompt | self.llm
        return chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)}).content

# ---------------------------------------------------------------------------
# 5. EXECUTION PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0
    )

    ticker = "RELIANCE"

    import time

    # Step A: Technical Analysis
    tech_df = load_technical_data(ticker_symbol=ticker)
    tech_agent = TechnicalAnalysisAgent(llm)
    print("=== TECHNICAL AGENT OUTPUT ===")
    print(tech_agent.analyze(ticker, tech_df.tail(30)))

    # PAUSE to avoid rate limiting
    print("\nWaiting 10 seconds for API rate limit cooldown...")
    time.sleep(10)

    # Step B: Sentiment Analysis
    news_df = load_sentiment_data(ticker_symbol=ticker)
    sentiment_agent = SentimentAnalysisAgent(llm)
    print("\n=== SENTIMENT AGENT OUTPUT ===")
    print(sentiment_agent.analyze(ticker, news_df.tail(5)))  # Reduced to 5 rows to save tokens