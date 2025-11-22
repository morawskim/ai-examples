from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template = '''
Current conversation: {history}
  Human: {question}
 AI:
'''

prompt = PromptTemplate(
 template = template,
 input_variables = ['question', 'history']
)

openai_model = ChatOpenAI(model="gpt-4.1-mini")
llm_chain = prompt | openai_model | StrOutputParser()
history = ''

while True:
    question = input("Question: ")
    if question == 'quit':
        break

    response = llm_chain.invoke({'question': question, 'history': history})
    history = response
    print(history)
