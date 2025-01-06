import streamlit as st
import os
# For PDF content extraction
from parsing import *
# For embedding and vector store
from embedding import *
from langchain_huggingface import HuggingFaceEmbeddings




uploaded_files = st.file_uploader("Upload Resume",accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        if uploaded_file.type == "application/pdf":
            # Save the file locally
            save_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)  # Ensure 'temp' directory exists
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract content from the PDF
            text = extract_pdf_content(save_path)
            # summarize relevant info from resume
            summary = summarize_resume(text)
            # Split and embed the resume
            split_and_embed_resume(summary)
            # Embed job description
            embed_job_description(job_description)

            # Search for similar resumes
            results = search_similar_resumes(job_description)

            for result in results:
                print(f"Resume Content: {result['resume_content']}")
                print(f"Similarity Score: {result['similarity_score']}\n")
                st.write(f"Resume Content: {result['resume_content']}")
                st.write(f"Similarity Score: {result['similarity_score']}\n")
            # Delete the file after processing
            if os.path.exists(save_path):
                os.remove(save_path)
        
        else:
            st.write("File not supported. Please upload a PDF file.")



