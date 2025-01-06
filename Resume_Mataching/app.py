import streamlit as st
import os
# For PDF content extraction
from parsing import *
# For embedding and vector store
from embedding import *
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(page_title="Resume Match with Job Description using NLP", page_icon="🦜")
st.title("🦜 Resume Match with Job Description using NLP")

# User inputs
job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resume", accept_multiple_files=True, type=["pdf"])

# Add a submit button
if st.button("Submit"):
    if not job_description.strip():
        st.error("Please enter a job description before submitting.")
    elif not uploaded_files:
        st.error("Please upload at least one resume before submitting.")
    else:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Display file details
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}

                if uploaded_file.type == "application/pdf":
                    # Save the file locally
                    save_path = os.path.join("temp", uploaded_file.name)
                    os.makedirs("temp", exist_ok=True)  # Ensure 'temp' directory exists
                    
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract content from the PDF
                    text = extract_pdf_content(save_path)
                    # Summarize relevant info from resume
                    summary = summarize_resume(text)
                    # Split and embed the resume
                    split_and_embed_resume(summary)
                    # Embed job description
                    embed_job_description(job_description)
                    
                    # Search for similar resumes
                    results = search_similar_resumes(job_description)

                    for result in results:
                        st.write(f"Resume Content: {result['resume_content']}")
                        st.write(f"Similarity Score: {result['similarity_score']}\n")
                    
                    # Delete the file after processing
                    if os.path.exists(save_path):
                        os.remove(save_path)
                else:
                    st.write("File not supported. Please upload a PDF file.")
