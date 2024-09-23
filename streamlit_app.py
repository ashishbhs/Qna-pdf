import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import re
import json
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            print(text)
    return text

# Function to split text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
        If the answer is not in the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.
        Context:\n{context}\n
        Question: {question}
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and query the vector store
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.info(response["output_text"])

# Helper function to parse JSON-like output
def parse_json_like(text):
    object_pattern = r'\{.*?\}'
    pair_pattern = r'"(.*?)":\s*"(.*?)"'
    objects = re.findall(object_pattern, text, re.DOTALL)
    dict_list = []
    for obj in objects:
        pairs = re.findall(pair_pattern, obj)
        d = {key: value for key, value in pairs}
        dict_list.append(d)
    return dict_list

# Function to generate MCQs from the PDF text
def generate_mcqs(text, number):
    response_schemas = [
        ResponseSchema(name="question", description="A multiple choice question generated from input text snippet."),
        ResponseSchema(name="option_1", description="First option for the multiple choice question. Use this format: 'a) option'"),
        ResponseSchema(name="option_2", description="Second option for the multiple choice question. Use this format: 'b) option'"),
        ResponseSchema(name="option_3", description="Third option for the multiple choice question. Use this format: 'c) option'"),
        ResponseSchema(name="option_4", description="Fourth option for the multiple choice question. Use this format: 'd) option'"),
        ResponseSchema(name="answer", description="Correct answer for the question. Use this format: 'd) option' or 'b) option', etc.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Define the prompt template
    prompt_template = """
        Given a text input, generate multiple choice questions (at least {number}) from it along with the correct answers.
        Question format:
        {format_instructions}
        Text: {user_prompt}
    """
    
    # Format the prompt with input text and number of MCQs
    prompt = prompt_template.format(
        user_prompt=text,
        number=number,
        format_instructions=format_instructions
    )
    
    # Directly pass the prompt to the model without using .to_messages()
    user_query_output = model(HumanMessage(content=prompt))
    
    output = user_query_output.content
    st.write(output)

    # Parse the output using the custom JSON-like parser
    mcqs = parse_json_like(output)
    
    return mcqs

# Function to initialize session state for quiz
def initialize_session_state():
    if 'form_count' not in st.session_state:
        st.session_state.form_count = 0
    if not hasattr(st.session_state, 'quiz_data') or not st.session_state.quiz_data:
        text = get_pdf_text(pdf_docs)
        st.session_state.quiz_data = generate_mcqs(text, number)

# Streamlit page setup
st.set_page_config("Chat with PDF")
st.header("Chat with PDF 📝✔️")

# PDF file uploader
with st.sidebar:
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

# Asking a question from PDF
user_question = st.text_input("##### Ask a Question from the PDF Files")
if user_question:
    user_input(user_question)

# Slider to select number of MCQs and button to generate them
number = st.slider("##### Select the number of MCQs", 5, 30, 5)
if st.button("Generate MCQs"):
    with st.spinner("Processing..."):
        initialize_session_state()
        quiz_data = st.session_state.quiz_data
        submitted_answers = [False] * len(quiz_data)
        
        for idx, mcq in enumerate(quiz_data, 1):
            st.markdown(f" **Question {idx}**:  {mcq['question']}")
            form = st.form(key=f"quiz_form_{st.session_state.form_count}_{idx}")
            user_choice = form.radio("Choose an answer:",
                                     [mcq["option_1"], mcq["option_2"], mcq["option_3"], mcq["option_4"]], index=None)
            submitted = form.form_submit_button("Submit your answer")
            submitted_answers[idx - 1] = submitted

            if submitted:
                if user_choice == mcq['answer']:
                    st.success(f"Question {idx}: Correct")
                else:
                    st.error(f"Question {idx}: Incorrect")