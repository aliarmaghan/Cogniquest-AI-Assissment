from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os, streamlit as st

# Initialize the embedding model
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to split and embed resumes
def split_and_embed_resume(resume_text, vectorstore_path="resume_faiss_index"):
   # Split the document into smaller chunks
   # text_splitter = RecursiveCharacterTextSplitter(
   #    chunk_size=500,  # Chunk size
   #    chunk_overlap=50  # Overlap between chunks
   # )
   # docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(resume_text)]
   
   # Create a document object for the resume
   docs = [Document(page_content=resume_text)]
   # Create embeddings and store in FAISS
   vectorstore = FAISS.from_documents(docs, embedding_model)
   vectorstore.save_local(vectorstore_path)  # Save the vectorstore to disk
   return vectorstore

# embed job descriptions and store them in the vector database
def embed_job_description(job_description, vectorstore_path="job_faiss_index"):
   # Create a document object for the job description
   job_doc = [Document(page_content=job_description)]
   
   # Embed and save to FAISS
   vectorstore = FAISS.from_documents(job_doc, embedding_model)
   vectorstore.save_local(vectorstore_path)  # Save the vectorstore to disk
   return vectorstore


# Function to search for similar resumes
def search_similar_resumes(job_description, resume_vectorstore_path="resume_faiss_index"):
   # Load the resume embeddings
   resume_vectorstore = FAISS.load_local(resume_vectorstore_path, embedding_model,allow_dangerous_deserialization=True)
   
   # Embed the job description
   job_embedding = embedding_model.embed_query(job_description)
   
   # Perform similarity search
   search_results = resume_vectorstore.similarity_search_with_score_by_vector(job_embedding)
   
   # Format results
   results = []
   for doc, score in search_results:
      results.append({"resume_content": doc.page_content, "similarity_score": score})
   return results
