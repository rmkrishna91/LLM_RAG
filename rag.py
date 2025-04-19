import getpass
import os
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
os.environ["COHERE_API_KEY"] = ""

st.sidebar.title('Custom GPT')
st.sidebar.button('New chat',type='primary')




prompt = ChatPromptTemplate.from_messages([
("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""),
])
llm = ChatCohere(model="command-r-plus",)
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vector_store = Chroma(embedding_function=embeddings)


file_path = '/Users/krishna/Downloads/resume_bosch.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
pages = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)


#Inititalizing chat history for streamli interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])



class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

user_ip = st.chat_input('say something')
if user_ip:
    with st.chat_message('user'):
        st.markdown(user_ip)
        st.session_state.messages.append({'role':'user','content':user_ip})


    result = graph.invoke({"question": user_ip})
    response = result['answer']

    with st.chat_message("assistant"):
    # Add assistant response to chat history
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


