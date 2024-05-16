import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.environ['GOOGLE_API_KEY']
os.getenv("GOOGLE_API_KEY")
def preprocess_text(text):
    # Remove HTML tagss
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def extract_questions(text):
    sentences = sent_tokenize(text)
    questions = [sentence.strip() for sentence in sentences if sentence.endswith('?')]
    return questions

def generate_responses(prompt_parts, model):
    response = model.generate_content(prompt_parts)
    return response.text

def main():
    st.title("FAQ Extraction Web App")

    # File Upload
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df = df['Transcript'].str.cat(sep='\n')

        # Preprocess the text
        cleaned_text = preprocess_text(df)

        # Initialize Google Gemini
        genai.configure(api_key=google_api_key)
        generation_config = { "temperature": 1, "top_p": 1, "top_k": 1, "max_output_tokens": 8192,}
        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
        ]

        # Create the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        # Topic Selection
        topic = st.selectbox("Select Topic", ["Ackumen Boiling Management (ABM)", "ACM", "User Management", "Connected Planning (CP)", "Process View (PV)", "Ackumen General", "Connected Lab", "Ackumen Orders", "ADE", "MCA"])
        prompt_parts = [cleaned_text, f"Retrieve the top 10 frequently asked questions (FAQs) as questions about {topic} from the given content, along with the respective frequency count of each question which is greated than 35. A question is an Frequently Asked Question only if it exceeds 35 in its occurance in the text. I need only questions and not answers"]

        # Generate content for selected topic
        response = generate_responses(prompt_parts, model)
        st.text(response)

        # Download Excel File
        if st.button("Download Excel File"):
            # Create Excel Writer
            excel_writer = pd.ExcelWriter("output.xlsx", engine="xlsxwriter")

            # Write Responses to Excel Sheets
            df_responses = pd.DataFrame({"Response": [response]})
            df_responses.to_excel(excel_writer, sheet_name=topic, index=False)

            # Save Excel File
            excel_writer.save()

            # Download Excel File
            st.download_button(label="Download Excel", data=open("output.xlsx", "rb").read(), file_name="output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
