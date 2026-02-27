from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY not set. Create a .env file with GROQ_API_KEY."
    )
print("LangSmith Key:", os.getenv("LANGCHAIN_API_KEY"))
print("Project:", os.getenv("LANGCHAIN_PROJECT")) 

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

#model = ChatGroq(model="openai/gpt-oss-120b")
model = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the AI and can you explaine the types of AI?"})
print(result)
