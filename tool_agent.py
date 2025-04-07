import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults

from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
#############
#Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper=WikipediaAPIWrapper(top_k_result=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun (api_wrapper=wikipedia_wrapper)

duck_wrapper=DuckDuckGoSearchResults(name="Search")


st.title("LangChain - Chat with search")

##Sidebar for settings
st.sidebar.title("settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key",type="password")

if "message" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi,I'm a Chatbot who can search the we.How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt:=st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[duck_wrapper,arxiv,wiki]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
