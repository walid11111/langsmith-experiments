import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Load environment variables
load_dotenv()

# Optional: Set LangSmith project name
os.environ["LANGCHAIN_PROJECT"] = "Langchain-Agenttt"

# ----------------------------
# 1️⃣ Search Tool
# ----------------------------
search_tool = DuckDuckGoSearchRun()

# ----------------------------
# 2️⃣ Weather Tool
# ----------------------------
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a given city using WeatherStack"""
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return "Weather API key missing."

    # ⚠ Free plan supports HTTP only
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"API Error: {data}"

    temp = data["current"]["temperature"]
    description = data["current"]["weather_descriptions"][0]

    return f"The current temperature in {city} is {temp}°C with {description}."

# ----------------------------
# 3️⃣ LLM (Updated Model ✅)
# ----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # ✅ Changed model
    temperature=0
)

# ----------------------------
# 4️⃣ Pull ReAct Prompt
# ----------------------------
prompt = hub.pull("hwchase17/react")

# ----------------------------
# 5️⃣ Create ReAct Agent
# ----------------------------
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# ----------------------------
# 6️⃣ Agent Executor
# ----------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)

# ----------------------------
# 7️⃣ Invoke Agent
# ----------------------------
response = agent_executor.invoke(
    {"input": "What is the current temperature of swat?"}
     #{"input": "who wrote thing and grow rich book?"}
      #{"input": "Pakistan news today and also what is the current weather of pakistan"}
)

print("\nFinal Output:")
print(response["output"])