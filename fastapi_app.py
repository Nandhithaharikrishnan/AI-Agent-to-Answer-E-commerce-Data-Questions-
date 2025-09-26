# pip install fastapi uvicorn langchain-ollama python-multipart plotly pandas
import os
import re
import sqlite3
import json
import traceback
from typing import Literal

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase

# ------------------------------------------------------------------
# Environment / Globals
# ------------------------------------------------------------------
load_dotenv()

DB_PATH = "data.db"
OLLAMA_URL = "http://127.0.0.1:11434"
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

gemini_llm = ChatGoogleGenerativeAI(api_key=GEMINI_KEY, model="gemini-2.5-flash")
ollama_llm = Ollama(base_url=OLLAMA_URL, model="qwen2.5:7b")

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="AI Data Query Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str
    visualize: bool = False
    provider: Literal["gemini", "ollama"] = "gemini"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def extract_sql(raw: str) -> str:
    """Extract a valid SQL SELECT statement from an LLM response."""
    if not raw or not raw.strip():
        raise ValueError("Empty SQL response from LLM")

    patterns = [
        r"```sql\s*(.*?)\s*```",
        r"```\s*(SELECT.*?)\s*```",
        r"(SELECT\s+.*?)(?:;|$)"
    ]
    for p in patterns:
        m = re.search(p, raw, re.DOTALL | re.IGNORECASE)
        if m:
            sql = m.group(1).strip().rstrip(';')
            if sql.lower().startswith("select"):
                return sql
    raise ValueError(f"Could not extract valid SQL from LLM response: {raw[:200]}...")

def answer(question: str, provider: str, visualize: bool):
    try:
        llm = gemini_llm if provider == "gemini" else ollama_llm
        chain = create_sql_query_chain(llm, db)

        raw_sql = chain.invoke({"question": question})
        sql = extract_sql(raw_sql)

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)

        chart_base64 = None
        if visualize and not df.empty:
            chart_base64 = make_chart(df, question)

        return {"sql": sql, "table": df.to_dict(orient="records"), "chart_base64": chart_base64}
    except Exception:
        traceback.print_exc()
        raise

def make_chart(df: pd.DataFrame, question: str) -> str:
    import base64, io
    numeric_cols = df.select_dtypes(include=["number"]).columns
    x_col = df.columns[0]
    y_col = numeric_cols[0] if numeric_cols.any() else df.columns[1]

    if df[x_col].dtype == "object" or len(df[x_col].unique()) < 20:
        fig = px.bar(df, x=x_col, y=y_col, title=f"Chart for: {question}")
    else:
        fig = px.line(df, x=x_col, y=y_col, title=f"Chart for: {question}")

    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
def health_check():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cur.fetchall()]
            sample_data = {}
            if tables:
                table = tables[0]
                cur.execute(f"SELECT * FROM {table} LIMIT 3;")
                rows = cur.fetchall()
                cur.execute(f"PRAGMA table_info({table});")
                cols = [c[1] for c in cur.fetchall()]
                sample_data = {"table": table, "columns": cols, "sample_rows": rows}

        return {
            "status": "healthy",
            "database": f"connected, tables: {tables}",
            "sample_data": sample_data,
            "gemini_api_key": "present" if GEMINI_KEY else "missing",
            "ollama_url": OLLAMA_URL,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/test-sql")
def test_sql():
    try:
        q = "Show me all data"
        chain = create_sql_query_chain(gemini_llm, db)
        raw_sql = chain.invoke({"question": q})
        return {"question": q, "raw_response": raw_sql, "extracted_sql": extract_sql(raw_sql)}
    except Exception:
        return {"error": traceback.format_exc()}

# -------- FIX: accept BOTH GET and POST ----------
@app.api_route("/query", methods=["GET", "POST"])
async def query_endpoint(request: Request):
    try:
        if request.method == "GET":
            question = request.query_params.get("question")
            visualize = request.query_params.get("visualize", "false").lower() == "true"
            provider = request.query_params.get("provider", "gemini")
        else:
            body = await request.json()
            question = body.get("question")
            visualize = body.get("visualize", False)
            provider = body.get("provider", "gemini")

        if not question:
            raise HTTPException(status_code=400, detail="Missing question")

        result = answer(question, provider, visualize)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static files (if you serve any images/HTML)
app.mount("/static", StaticFiles(directory="."), name="static")
