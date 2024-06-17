__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 환경 변수 설정
api_key = 'AIzaSyBZ1i_UKI0FqchGh3nOombiVheJ-HDL8_Q'
os.environ["GOOGLE_API_KEY"] = api_key
os.environ["USER_AGENT"] = "my_user_agent"

#제목
st.title("세아제강 AI Service")
st.write("---")

# 샘플 파일 로드
loader = PyPDFLoader('secret.pdf')
documents = loader.load_and_split()

text_splitter = CharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(documents)

# Embeddings 설정 (Google Generative AI 사용)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector DB 설정 (Chroma 사용)
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 질문 입력 UI
st.header("세아 AI에게 질문해보세요!!")
query = st.text_input('질문을 입력하세요')

if st.button('질문하기'):
    with st.spinner('Wait for it...'):
        if query:
            # 프롬프트 템플릿 설정
            system_template = """
            Use the following pieces of context to answer the users question shortly.
            Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
            If you don't know the answer, just say that "I don't know", don't try to make up an answer.
            ----------------
            {summaries}
            You MUST answer in Korean and in Markdown format:"""
            messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            prompt = ChatPromptTemplate.from_messages(messages)

            # 모델 학습 설정 (Google Generative AI 사용)
            chain_type_kwargs = {"prompt": prompt}
            llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                         temperature=0.4,
                                         convert_system_message_to_human=True)
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )

            # 결과 확인
            result = chain.invoke(query)
            st.write(result['answer'])

