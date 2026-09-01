import os
import pandas as pd

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
# Set your Groq API key
os.environ["GROQ_API_KEY"] = "YOUR_NEW_GROQ_API_KEY"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"]
)
FUNDAMENTAL_FILE = "/content/fundamentals.csv"

df = pd.read_csv(FUNDAMENTAL_FILE)

print(df.head())
print(df.columns.tolist())
def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        ticker = str(row.get("Stock", "")).upper()

        text = f"""
Company: {ticker}

Market Capitalization: {row.get("Market Cap", "N/A")}

Revenue: {row.get("Revenue", "N/A")}

Net Profit: {row.get("Net Profit", "N/A")}

EPS: {row.get("EPS", "N/A")}

PE Ratio: {row.get("PE Ratio", "N/A")}

ROE: {row.get("ROE", "N/A")}

Debt to Equity: {row.get("Debt to Equity", "N/A")}

Dividend Yield: {row.get("Dividend Yield", "N/A")}
"""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "ticker": ticker,
                    "source": "fundamental_dataset"
                }
            )
        )

    return documents


documents = create_documents(df)

print(f"Created {len(documents)} documents.")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

print("FAISS vector database created successfully.")
class FundamentalRAGAgent:

    def __init__(self, llm, vectorstore):

        self.llm = llm
        self.vectorstore = vectorstore

        self.prompt = ChatPromptTemplate.from_template("""
You are a Fundamental Analysis AI Agent specializing
in Indian equity markets.

Your task is to evaluate the fundamental health of a company
using ONLY the retrieved company information provided below.

Company requested:
{ticker}

Retrieved fundamental information:
{context}

Analyze:

1. Revenue and profitability
2. EPS
3. P/E valuation
4. ROE
5. Debt-to-equity
6. Dividend yield
7. Overall financial strength

Do not invent missing values.

Return ONLY valid JSON in this format:

{{
    "fundamental_signal": "BULLISH",
    "confidence": 0.85,
    "reasoning": "Brief explanation based on the retrieved fundamentals."
}}

fundamental_signal must be one of:

BULLISH
BEARISH
NEUTRAL

confidence must be between 0.0 and 1.0.
""")

    def analyze(self, ticker):

        ticker = ticker.upper().strip()

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(
            ticker,
            k=3
        )

        if not docs:

            return {
                "fundamental_signal": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "No fundamental information was found."
            }

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        chain = self.prompt | self.llm

        response = chain.invoke({
            "ticker": ticker,
            "context": context
        })

        return response.content
    fundamental_agent = FundamentalRAGAgent(
    llm=llm,
    vectorstore=vectorstore
)

result = fundamental_agent.analyze("RELIANCE")

print("=== FUNDAMENTAL AGENT ===")
print(result)