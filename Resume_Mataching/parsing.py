import streamlit as st
import pdfplumber
from docling.document_converter import DocumentConverter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


## Get the Groq API Key 
import os
from dotenv import load_dotenv
load_dotenv()


# Extract text from a PDF file
def extract_text_from_pdf(file_path):
   """Extracts text from a PDF file."""
   with pdfplumber.open(file_path) as pdf:
      text = ""
      for page in pdf.pages:
         text += page.extract_text() + "\n"
      return text

def extract_pdf_content(file) -> str:
   """
   Extract structured content from PDF using Docling library
   
   Args:
      file_path (str): Path to the PDF file
      
   Returns:
      str: Extracted content in markdown format
   """
   converter = DocumentConverter()
   result = converter.convert(file)
   res = result.document.export_to_markdown()
   
   return res


# Function to summarize a resume
def summarize_resume(text):

   docs = [Document(page_content=text)]
   llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
                model="gemma2-9b-it",
                max_tokens=1000,
                temperature=0.7)
   system_prompt = PromptTemplate(
   input_variables=["text"],  # Define the input variable(s) expected in the template
   template="""
   You are an AI assistant designed to analyze resumes and create structured summaries. Your task is to extract and summarize the key details from a resume. Follow this format:

   1. Experience Summary:
      - Summarize the total years of experience and detail the years of experience in each industry/domain mentioned in the resume.

   2. Project Summary:
      - Summarize the total number of projects. Mention their focus areas, technologies used, or notable outcomes.

   3. Skills:
      - Extract and list all technical and soft skills mentioned in the resume.

   4. Certifications:
      - Extract and list all certifications mentioned in the resume, along with the institution and date (if available).

   5. Education:
      - Extract and summarize the candidate’s educational qualifications, including degrees, institutions, and graduation years.
   6. Achievements:
      - Extract and summarize any notable achievements, awards, recognitions, or accolades mentioned in the resume.

   Provide the summary in clear, concise paragraphs while maintaining the structure outlined above : {text}.
   """
   )
   chain = load_summarize_chain(llm,chain_type="stuff",prompt=system_prompt)
   output_summary = chain.run(docs) 

   return output_summary








