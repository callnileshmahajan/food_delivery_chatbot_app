# agent.py
import json
import os
import pandas as pd
import sqlite3

from langchain.agents import create_sql_agent, initialize_agent # Added initialize_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import load_tools
from langchain.agents import Tool
from langchain_groq import ChatGroq  # Import Groq LLM
from pydantic import BaseModel, Field, ValidationError

from ddgs import DDGS
from typing import List, Optional, Dict
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -----------------------------
# Helpers
# -----------------------------
def _extract_order_id(text: str) -> Optional[str]:
    """
    Extracts order id like O12486 from user text.
    Adjust regex if your IDs differ.
    """
    match = re.search(r"O\d+", text, re.IGNORECASE)
    return match.group(0) if match else None


def _strip_code_fences(text: str) -> str:
    """
    Removes ```json ... ``` fences if the model wraps JSON.
    """
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _agent_output_to_text(result: Union[str, Dict[str, Any]]) -> str:
    """
    create_sql_agent().invoke() can return dict or string depending on version/settings.
    This safely extracts the final output text.
    """
    if isinstance(result, dict):
        # typical shape: {'input': '...', 'output': '...'}
        return result.get("output", "")
    return result


def _parse_json_if_possible(text: str) -> Union[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Try to parse JSON; if it fails, return the original text.
    """
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        return text


# -----------------------------
# LLM + DB + SQL Agent factory
# -----------------------------
def build_sql_agent() -> Any:
    """
    Creates and returns a LangChain SQL agent executor (db_agent).
    Uses env vars so it works in Hugging Face Spaces.
    """
    # 1) LLM
    MODEL = "llama-3.3-70b-versatile"
    # Get the API key from Colab secrets
    groq_api_key = os.getenv('GROQ_API_KEY')

    #llama-3.3-70b-versatile
    # Low creativity (deterministic) LLM
    llm = ChatGroq(
        model=MODEL, #Call the Groq model
        temperature=0,
        groq_api_key=groq_api_key,
        max_retries=2,
        timeout=30
    )

    LLM_CONFIG = {"provider": "Groq", "model": MODEL, "temperature": 0, "timeout": 30, "max_retries": 2}
    print(LLM_CONFIG)

    # 2) Database connection
    # Option A: SQLite shipped in repo, e.g. food_delivery.db
    # DATABASE_URI="sqlite:///food_delivery.db"
    # Option B: hosted DB (Postgres/MySQL/etc) via URI in Secrets
    file = "customer_orders.db"
    db = SQLDatabase.from_uri(f"sqlite:///{file}")

    # 3) System message for SQL agent
    # Important: make SQL agent NOT output "Action: None"
    # If missing order_id, it should produce a Final Answer asking for it.
    system_message = """
        You are an SQL assistant for a food delivery database.

        RULES:
        - Use ONLY read-only queries (SELECT). Never use INSERT, UPDATE, DELETE, DROP, ALTER.
        - Use the SQL tools available via the toolkit to query the database.
        - If the user request requires an order id and it is missing, respond ONLY with:
          Final Answer: Please provide your Order ID (e.g., O12486).
        - NEVER output 'Action: None'.

        OUTPUT:
        - When you query order data, return ONLY valid JSON (no markdown, no explanations).
        - JSON must be either:
          (a) a list of objects (rows), or
          (b) a single object.
        - If nothing found, return [].
        """.strip()

    #Initialize the toolkit with customer database and the LLM
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    #Create the SQL agent with the system message
    db_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        max_iterations=6,
        max_execution_time=60,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        system_message=SystemMessage(system_message)
    )

    return db_agent


# Build once (Streamlit reruns script; we want reuse)
DB_AGENT = build_sql_agent()


# -----------------------------
# Tools for Chat Agent (rubric)
# -----------------------------
def order_query_tool(order_context: Dict[str, Any], user_query: str) -> str:
    """
    Takes order context from SQL agent (dict) and generates raw response.
    Keep it factual; no extra niceties.
    """
    if not isinstance(order_context, dict) or not order_context:
        return "I could not find order details for the provided information."

    order_id = order_context.get("order_id") or order_context.get("Order_ID") or order_context.get("ORDER_ID")
    status = order_context.get("order_status") or order_context.get("status") or order_context.get("Order_Status") or order_context.get("ORDER_STATUS")
    time_ = order_context.get("order_time") or order_context.get("time") or order_context.get("Order_Time") or order_context.get("ORDER_TIME")
    items = order_context.get("items") or order_context.get("Items") or order_context.get("ITEMS")

    # Normalize items display
    if isinstance(items, list):
        items_text = ", ".join([str(x) for x in items])
    elif items is None:
        items_text = "Not available"
    else:
        items_text = str(items)

    lines = []
    if order_id: lines.append(f"Order ID: {order_id}")
    if status:  lines.append(f"Order Status: {status}")
    if time_:   lines.append(f"Order Time: {time_}")
    if items_text: lines.append(f"Items: {items_text}")

    return "\n".join(lines) if lines else "Order details are not available."


def answer_tool(raw_response: str) -> str:
    """
    Refines raw response into polite, formal reply.
    """
    return (
        "Thank you for your query.\n\n"
        "Please find the requested order information below:\n\n"
        f"{raw_response}\n\n"
        "If you need any further assistance, please let me know."
    )


def combined_order_tool(order_context: Dict[str, Any], user_query: str) -> str:
    raw = order_query_tool(order_context, user_query)
    return answer_tool(raw)


# -----------------------------
# Main Chat Agent entry point
# -----------------------------
def chat_agent(
    user_query: str,
    conversation_state: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (final_answer, updated_state)
    State is used to remember last_order_id for follow-up questions.
    """
    state = conversation_state or {}
    last_order_id = state.get("last_order_id")

    # 1) Find order_id from user query or state
    order_id = _extract_order_id(user_query) or last_order_id

    # If user asks something requiring order id but none found:
    needs_id_keywords = ["order", "delivery", "status", "where", "location", "track"]
    if (order_id is None) and any(k in user_query.lower() for k in needs_id_keywords):
        return ("Please provide your Order ID (e.g., O12486) so I can check the order details.", state)

    # 2) If we found an order_id, store it in state
    if order_id:
        state["last_order_id"] = order_id

    # 3) Ask SQL agent to fetch *structured* order data (JSON)
    # We include order_id explicitly to reduce model confusion.
    sql_question = (
        f"Fetch the latest details for order_id='{order_id}'. "
        f"Return only JSON with fields like order_id, order_time, order_status, items, delivery_location if available."
    ) if order_id else user_query

    sql_result = DB_AGENT.invoke({"input": sql_question})
    output_text = _agent_output_to_text(sql_result)

    # If SQL agent returned a ReAct-compliant Final Answer asking for ID:
    if isinstance(output_text, str) and output_text.lower().startswith("final answer:"):
        msg = output_text.split("Final Answer:", 1)[-1].strip()
        return (msg, state)

    # 4) Parse JSON if present
    parsed = _parse_json_if_possible(output_text)

    # If parsing failed and we got plain text, just return a polite version
    if isinstance(parsed, str):
        # This could happen if the SQL agent produced natural language.
        # We still keep UX natural.
        return (answer_tool(parsed), state)

    # 5) Choose a single order_context dict
    if isinstance(parsed, list):
        order_context = parsed[0] if parsed else {}
    else:
        order_context = parsed

    # 6) Generate final natural language response
    final = combined_order_tool(order_context, user_query)
    return (final, state)
