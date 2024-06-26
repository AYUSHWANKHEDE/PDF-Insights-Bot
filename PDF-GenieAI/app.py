import os
import time
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI API key
google_api_key = os.getenv("GEMINI_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

# Initialize Flask app
app = Flask(__name__)

# Utility functions
def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_vector_store(text_chunks, retries=3, delay=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    for attempt in range(retries):
        try:
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""

def setup_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def use_gemini_chat(user_question,content):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[
            {"role": "user",
      "parts": [
        f"remember this content - {content}",
      ],}
        ]
    )

    response = chat_session.send_message(user_question)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_docs = request.files.getlist('pdf_docs')
        user_question = request.form['user_question']

        try:
            pdf_text = get_pdf_text(pdf_docs)
            if not pdf_text.strip():  # Check if the extracted text is empty
                return render_template('error.html', error="The uploaded PDF is empty or could not be read.")
            
            text_chunks = get_text_chunks(pdf_text)
            vector_store = initialize_vector_store(text_chunks)

            docs = vector_store.similarity_search(user_question)
            conversational_chain = setup_conversational_chain()
            response = conversational_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            answer = response["output_text"]
            if "answer is not available in the context" in answer.lower():
                answer = use_gemini_chat(user_question)

            return render_template('result.html', question=user_question, response=answer)

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    pdf_docs = request.files.getlist('pdf_docs')

    try:
        pdf_text = get_pdf_text(pdf_docs)
        if not pdf_text.strip():  # Check if the extracted text is empty
            return render_template('error.html', error="The uploaded PDF is empty or could not be read.")
        
        summary_question = "Provide a summary of the PDF."
        text_chunks = get_text_chunks(pdf_text)
        vector_store = initialize_vector_store(text_chunks)

        docs = vector_store.similarity_search(summary_question)
        conversational_chain = setup_conversational_chain()
        response = conversational_chain({"input_documents": docs, "question": summary_question}, return_only_outputs=True)
        
        summary = response["output_text"]
        if "answer is not available in the context" in summary.lower():
            summary = use_gemini_chat(summary_question,pdf_text)

        return render_template('result.html', question=summary_question, response=summary)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
