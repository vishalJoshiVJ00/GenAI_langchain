from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system','you are the best {domain} expert'),
    ('human', 'hello, exaplain the topic {topic}')
])

prompt = chat_template.invoke({'domain':'web', 'topic' : 'html'})

print(prompt)