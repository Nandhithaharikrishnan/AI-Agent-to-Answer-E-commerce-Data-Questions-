import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
import matplotlib.pyplot as plt
import plotly.express as px
import re
import json

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Set up database
DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")

# Create SQL query chain
sql_chain = create_sql_query_chain(llm, db)


def extract_sql(query_output: str) -> str:
    """Extract actual SQL from the LLM response."""
    match = re.search(r"```sql\n(.*?)\n```", query_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = query_output.strip().splitlines()
    for line in lines:
        if line.strip().lower().startswith("sqlquery:"):
            return line.split(":", 1)[-1].strip()

    if query_output.strip().lower().startswith("select"):
        return query_output.strip()

    return query_output.strip()


def get_chart_specs(question: str, df: pd.DataFrame) -> dict:
    """Use an LLM to determine the best chart to build."""
    prompt = f"""
    Given a user's question and a resulting data table, your task is to recommend the best Plotly chart and its parameters.
    The user's question was: "{question}"
    The data table has the following columns: {list(df.columns)}
    Respond with a JSON object with the following keys:
    - "chart_type": string (e.g., "bar", "line", "scatter", "pie").
    - "x": string (the column name for the x-axis).
    - "y": string or list of strings (the column name(s) for the y-axis).
    - "title": string (a descriptive title for the chart).
    - "color": string (optional, the column to use for color encoding).
    Example:
    {{
        "chart_type": "bar",
        "x": "item_id",
        "y": ["total_sales", "ad_sales"],
        "title": "Total Sales vs. Ad Sales for Top Products",
        "color": null
    }}
    Based on the user's question and the data columns, what is the best chart to create?
    """

    try:
        response = llm.invoke(prompt)
        json_str = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to get chart specs from LLM: {e}")
        return None


def generate_chart(df: pd.DataFrame, question: str, specs: dict = None):
    """Generate and save a chart from the dataframe."""
    if df.empty:
        print("Not enough data to generate a chart.")
        return

    if not specs:
        print("No chart specifications provided, attempting a simple chart.")
        if len(df.columns) < 2:
            return
        specs = {
            "chart_type": "bar",
            "x": df.columns[0],
            "y": df.columns[1],
            "title": f"Visualization for: {question}",
            "color": None
        }

    try:
        chart_type = specs.get("chart_type", "bar")
        x = specs.get('x')
        y = specs.get('y')
        title = specs.get('title')
        color = specs.get('color')

        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title, color=color)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title, color=color)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title)
        else:
            print(f"Unsupported chart type: {chart_type}. Defaulting to bar chart.")
            fig = px.bar(df, x=x, y=y, title=title)

        chart_path = 'chart.png'
        fig.write_image(chart_path)
        print(f"\nChart saved to {os.path.abspath(chart_path)}")

    except Exception as e:
        print(f"\nFailed to generate chart with Plotly: {e}")


def answer_question(question: str, visualize: bool = False):
    try:
        query_prompt = question
        if visualize:
            query_prompt += " (generate a query that is good for a chart, e.g., with labels and values)"

        raw_sql_output = sql_chain.invoke({"question": query_prompt})
        print("\nRaw SQL Output:\n", raw_sql_output)
        sql_query = extract_sql(raw_sql_output)
        print("\nParsed SQL Query:\n", sql_query)
    except Exception as e:
        return f"Failed to generate SQL query: {e}"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return f"Failed to execute SQL query: {e}"

    if df.empty:
        return "Query executed, but no results found."

    if visualize:
        chart_specs = get_chart_specs(question, df)
        generate_chart(df, question, chart_specs)

    return df.to_string(index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Data Agent")
    parser.add_argument('question', type=str, help='Question to ask about the data')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()
    result = answer_question(args.question, visualize=args.visualize)
    print("\nAnswer:\n", result)