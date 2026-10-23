
import os
import requests
import certifi
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor


# -----------------------------
# 1. App configuration
# -----------------------------
st.set_page_config(
    page_title="AI Search & Weather Agent",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ AI Search & Weather Agent")
st.write(
    "Ask questions, search the web, and retrieve current weather "
    "using Gemini, Tavily, and WeatherAPI."
)

# Load .env from the folder containing app.py
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()


# -----------------------------
# 2. Validate API keys
# -----------------------------
missing_keys = []

if not GEMINI_API_KEY:
    missing_keys.append("GEMINI_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not WEATHER_API_KEY:
    missing_keys.append("WEATHER_API_KEY")

if missing_keys:
    st.error(
        "Missing API keys in your .env file: "
        + ", ".join(missing_keys)
    )
    st.stop()


# -----------------------------
# 3. Define weather tool
# -----------------------------
@tool
def get_weather_data(city: str) -> str:
    """Get the current temperature and humidity for a city."""

    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={
                "key": WEATHER_API_KEY,
                "q": city
            },
            timeout=20
        )

        response.raise_for_status()
        data = response.json()

        current = data.get("current")
        location = data.get("location")

        if not current or not location:
            return f"Weather data is unavailable for {city}."

        return (
            f"City: {location['name']}, {location['country']}\n"
            f"Temperature: {current['temp_c']}°C\n"
            f"Condition: {current['condition']['text']}\n"
            f"Humidity: {current['humidity']}%"
        )

    except requests.RequestException as exc:
        return f"Weather API request failed: {exc}"
    except (KeyError, ValueError):
        return f"Could not interpret weather data for {city}."


# -----------------------------
# 4. Configure Gemini LLM
# -----------------------------
@st.cache_resource
def create_agent_executor():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    search_tool = TavilySearchResults(
        max_results=3,
        tavily_api_key=TAVILY_API_KEY
    )

    tools = [search_tool, get_weather_data]

    prompt = PromptTemplate.from_template("""
Answer the user's question using the available tools.

Available tools:
{tools}

Follow this format:
Question: the input question
Thought: decide what information is needed
Action: one of [{tool_names}]
Action Input: the tool input
Observation: the tool result
Thought: I now know the final answer
Final Answer: the answer to the user

Question: {input}
Thought:{agent_scratchpad}
""")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6
    )


# -----------------------------
# 5. Streamlit user interface
# -----------------------------
st.subheader("Ask your AI agent")

question = st.text_area(
    "Enter your question",
    value="What is the capital of India, and what is its current weather?",
    height=100,
    placeholder="Example: Find Hyderabad's current temperature."
)

if st.button("Run AI Agent", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Gemini is working with your tools..."):
                executor = create_agent_executor()
                result = executor.invoke({"input": question.strip()})

            st.subheader("Agent Response")
            st.markdown(result.get("output", "No response returned."))

        except Exception as exc:
            st.error("The agent could not complete the request.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.caption(
                "Check your API keys, model availability, quotas, "
                "and LangChain tool configuration."
            )

st.divider()
st.caption("Powered by Google Gemini · Tavily Search · WeatherAPI")
