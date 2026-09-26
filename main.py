from __future__ import annotations

import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from prompt import SUPERVISOR_PROMPT, SYSTEM_PROMPT_PQ, SYSTEM_PROMPT_SUPPORT
from langchain.agents.middleware import SummarizationMiddleware
from mcp_clients import database
from web_scraper import web_crawler
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
from web_scraper import web_crawler
from kb_tool import questohive_db

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log"),
    ],
)
logger = logging.getLogger("QuestoHive")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    LLM_MODEL = "deepseek-v4-flash"
    LLM_TEMPERATURE = 0
    REQUEST_TIMEOUT = 120


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment() -> None:
    required = [
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Cannot start — missing environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file."
        )

    
# ============================================================================
# LLM FACTORY
# ============================================================================

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=Config.LLM_TEMPERATURE,
        model=Config.LLM_MODEL,
        max_completion_tokens=5000,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=Config.REQUEST_TIMEOUT,
        streaming=True,
    )




support_agent = create_agent(
        model=make_llm(),
        tools=[web_crawler, questohive_db],
      #  checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT_SUPPORT,
        name="Support Agent",)
@tool(
    "Customer_Support_agent",
    description="Use this agent to get the answer to customer related questions about questohive",
)
async def Customer_Support_agent(query: str) -> str:
    """Used to get customer support information."""
    result = await support_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Could not get response at the moment"


pq_agent = create_agent(
        model=make_llm(),
        tools=[database],
        #checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT_PQ,
        name="PQ Agent",
    )

@tool(
    "Past_Questions_Fetcher",
    description="Use this tool to fetch past questions from the database.",
)
async def Past_Questions_Fetcher(query: str) -> str:
    """Used to get pastquestions from the database."""
    result = await pq_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Could not get response at the moment"



supervisor = create_agent(
        model=make_llm(),
        tools=[Past_Questions_Fetcher, Customer_Support_agent],
       # checkpointer=InMemorySaver(),
        system_prompt=SUPERVISOR_PROMPT,
        name="Supervisor Agent",
        middleware=[
            SummarizationMiddleware(
                model="deepseek-v4-flash",
                trigger=("tokens", 5000),
                keep=("messages", 20),
            ),
        ],
    )
    
agent = supervisor
