
# PDF-QA-GenieAI

PDF-QA-Genie is a Flask web application that allows users to upload PDF documents and ask questions based on the content of those documents. The application uses Google Generative AI to generate detailed answers to the user's questions and provides a summary of the PDF content.

## Features

- Upload multiple PDF documents.
- Extract text from PDF documents.
- Use Google Generative AI to generate embeddings and answers.
- Perform similarity search using FAISS.
- Provide detailed answers to user questions based on the PDF content.
- Generate a summary of the uploaded PDF content.

## Setup

### Prerequisites

- Python 3.7+
- Flask
- PyPDF2
- Langchain
- langchain_google_genai
- langchain_community
- FAISS
- dotenv

### Installation

1. Clone the repository:

```sh
git clone https://github.com/AYUSHWANKHEDE/PDF-QA-Genie.git
cd PDF-QA-Genie


