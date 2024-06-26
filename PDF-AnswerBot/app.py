import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pdfplumber
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini AI SDK with API key
genai.configure(api_key="AIzaSyDOeIByc7XQcebIlwqfVJvZHXNalrnyTPw")

# Replace with your Gemini AI endpoint
GEMINI_API_ENDPOINT = 'https://api.gemini.ai/ask'

# Configuration for file upload
UPLOAD_FOLDER = 'PDF-AnswerBot\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Error: No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "Error: No selected file", 400

    if not allowed_file(file.filename):
        return "Error: Invalid file type. Only PDF files are allowed.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Read PDF content
        with pdfplumber.open(file_path) as pdf:
            text = "".join([page.extract_text() for page in pdf.pages])

        # Save the extracted text to a text file
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        return render_template('ask_question.html', text_file_path=text_file_path)

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/ask', methods=['POST'])
def ask_question():
    text_file_path = request.form.get('text_file_path')
    question = request.form.get('question')

    if not text_file_path or not os.path.exists(text_file_path):
        return "Error: Text file not found.", 400

    if not question:
        return "Error: No question provided.", 400

    try:
        # Read text from the saved text file
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            text = text_file.read()

        # Initialize the Gemini AI model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # Start chat session with history and send question
        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [f"{text}\n"]}]
        )
        response = chat_session.send_message(question)
        answer = response.text.strip()

        return render_template('ans.html', answer=answer)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
