import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

os.environ['LANGUAGE_PROJECT']='Sequential-app'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {                # this code is just when you want to add extra infromation and this data show in langsmith to more understanding
    'tags' : ['Report generation','Summery'],
    'metadada' : {'model': 'gpt os 20b', 'parser': 'stroutparser'}
}

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
