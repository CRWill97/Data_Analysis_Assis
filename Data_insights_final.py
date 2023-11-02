#!/usr/bin/env python
# coding: utf-8

# In[74]:


#import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import tabulate
import pyarrow
import openai

#langchain_experimental.agents.create_pandas_dataframe_agent
# In[75]:


import streamlit as st


# In[76]:


openai_api_key = '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'


# In[77]:


st.set_page_config(page_title='🤖 Data Analysis Assistant')
st.title('🤖 Data Analysis Assistant')
st.header('To use this app you will need to have an OpenAI API key ')

# In[78]:


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df


# In[79]:


def generate_response(csv_file, input_query):
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    response = agent.run(input_query)
    return st.success(response)


# In[80]:


uploaded_file = st.file_uploader('Upload a CSV file', type = ['csv'])
question_list = [
    'What are the basic characteristics of my dataset?',
    'What is the overall structure of my dataset?',
    'What patterns exist in the data?',
    'Are there any outliers present?',
    'What are the missing values in the data set?',
    'Is there any correlation between variables?',
    'Are there any discrepancies between observed values and expected values?',
    'Do I need to transform any variables before analysis?',
    'Other:']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not(uploaded_file and query_text))


# In[81]:


if query_text == 'Other:':
    query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled = not uploaded_file)
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API Key!')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
    st.header('Output')
    generate_response(uploaded_file, query_text)


# In[ ]:





# In[ ]:




