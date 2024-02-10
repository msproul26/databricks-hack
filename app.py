from io import StringIO
import databricks
from langchain_community.retrievers import AmazonKendraRetriever
from langchain.prompts import PromptTemplate
from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from botocore.config import Config
from dotenv import load_dotenv
from databricks.sdk import AccountClient
import pandas as pd
import streamlit as st
import os
import boto3
import extract_msg
import mlflow.deployments
import json
import re
load_dotenv()

index_name = "hackathon_data_science_workspace.default.enronsentimentvector"
host = "https://dbc-689b2fd3-da3e.cloud.databricks.com"
VECTOR_SEARCH_ENDPOINT_NAME = "embeddings"
global_last_reply = ""



def format_prompt(query):
    return f"\n\nHuman: {query} \n\nAssistant:"

def prompttemplate(template, input_variables):
    def generate_prompt(context, question):
        if len(input_variables) != 2:
            raise ValueError("Input variables must be exactly two: context and question")

        # Replace placeholders in the template with provided values
        prompt_text = template.format(context=context, question=question)

        return f"\n\nHuman: {prompt_text} "

    return generate_prompt

def custom_parse_function(value):
    sentiment="0"
    m="0"
    pattern = r"sentiment\s*=\s*(-?\d+)"
    match = re.search(pattern, str(value))

    # Extract the value of sentiment if there is a match
    if match:
        sentiment = match.group(1)
        m="1" 
    return int(sentiment)

st.set_page_config(
  page_title="Email Analyzer",
  page_icon=":smiley:",
  layout="wide"
)

colimg, _ = st.columns([1, 2])
company_image = "Logo_Horz_Full Color.svg" 
with colimg:
    st.image(company_image, use_column_width=True)


st.title("Email Analyzer")


#Connect to Databricks using MLFLOW
dbclient = mlflow.deployments.get_deploy_client("databricks")
embedding_model = DatabricksEmbeddings(endpoint="embeddings")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host)
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore

@st.cache_data()
def get_sentiment(files):
    msg = extract_msg.Message(files[0])
    st.write(msg.body)
    sentimentprompt = format_prompt("Please give me a numeric sentiment score between -5 and 5, for the last email in this email chain. Return only the number in the format sentiment=score.: "+msg.body)
    sentimentscore = dbclient.predict(
    endpoint="bedrock_claude",
    inputs={
        "prompt": sentimentprompt,
        "max_tokens": 4000
    }
    )
    return sentimentscore

@st.cache_data()    
def get_email_reply(files):
    msg = extract_msg.Message(files[0])
    vsc = VectorSearchClient()
    results = vsc.get_index("embeddings", "hackathon_data_science_workspace.default.enronsentimentvector").similarity_search(
        query_text=msg.body,
        columns=["sentiment_value", "content"],
        num_results=10)
    
    filtered_results = [item for item in results["result"]["data_array"] if item[0][0] != '-' and float(item[0][0]) <= score+2]
    

    TEMPLATE = """You are an assistant for Outlook users. You are responding to the last email in the email chain provided at the end of this message. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    Use the following pieces of context to generate a response as a reply to the email at the end:
    {context}
     Here is the email chain you will generate a response for. Only reply to the last email in the chain. You are the recipent of the last email. If there is no to and from use the most recept message to reply : {question}
    """    
    prompt1 = prompttemplate(template=TEMPLATE, input_variables=["context", "question"])
    context_data = filtered_results
    question_data = msg.body
    formatted_prompt = prompt1(context=context_data, question=question_data)
    
    myllmcustom = dbclient.predict(
    endpoint="bedrock_claude",
    inputs={
        "prompt": formatted_prompt,
        "max_tokens": 4000
    }
    )
    return myllmcustom

files = st.file_uploader("Upload Email or Email Chain for Assistance.",type=['msg'], accept_multiple_files=True)
if files:
    sentimentscore = get_sentiment(files)
    
    score = custom_parse_function(sentimentscore["choices"][0]["text"])
    
    myllmcustom = get_email_reply(files)
    
    st.markdown("### Recommended Response:")
    st.write(myllmcustom["choices"][0]["text"])
    global_last_reply = myllmcustom["choices"][0]["text"]

query = st.text_input("Is there any additional information you would like to include in this response?")
def run_customization():
    with st.spinner("Building response..."):
        message = "Fix the following email: {context} Fix this email by applying the following notes: {question} "
        prompt2 = prompttemplate(template=message, input_variables=["context", "question"])
        context_data = global_last_reply
        question_data = query
        formatted_prompt = prompt2(context=context_data, question=question_data)
        
        updatedreply = dbclient.predict(
            endpoint="bedrock_claude",
            inputs={
                "prompt": formatted_prompt,
                "max_tokens": 4000
            }
            )
        return updatedreply
    
if st.button("Customize"):
    updatedresponse = run_customization()

    st.markdown("### Updated Email:")
    st.write(updatedresponse["choices"][0]["text"])


