
from json import tool
import os
import certifi
import requests



from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent,AgentExecutor

#---- errors ----
# pip install --upgrade pip setuptools wheel
# pip install --upgrade langchain
# pip install --only-binary=:all: cryptography




load_dotenv(override=True)
openai_api_key = os.getenv("OPEN_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY").strip()
hf_token = os.getenv("HF_TOKEN").strip()
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
WEATHER_API_KEY=os.getenv("WEATHER_API_KEY", "").strip()
os.environ["SSL_CERT_FILE"] = certifi.where()


search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=gemini_api_key,
    temperature=0
)



prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to these tools:

{tools}

Use this format:

Question: the input question
Thought: your reasoning about what to do
Action: the tool to use, must be one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Question: {input}
Thought:{agent_scratchpad}""")
#prompt

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
       
        f"https://api.weatherapi.com/v1/current.json?"
        f"key={WEATHER_API_KEY}&q={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"temp_c: {data['current']['temp_c']}°C\n"
        #f"condition: {data['current']['condition']}\n"
        f"humidity: {data['current']['humidity']}%"
    )
tools = [search_tool, get_weather_data]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

result = agent_executor.invoke({
    "input": "What is the capital of India?"
    "and then find its current weather."
})


