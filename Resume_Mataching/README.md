
# Resume Matching System Using NLP

## Overview
This project aims to streamline the recruitment process by building an intelligent system that matches candidate resumes with the most suitable job postings. The system leverages Natural Language Processing (NLP) to evaluate compatibility based on skills, experience, qualifications, and other pertinent factors. 

The solution extracts structured information from resumes, embeds job descriptions and resume texts, and evaluates the similarity to provide ranked matches.

## Features
- **Resume Parsing**: Extracts and summarizes key information (skills, experience, projects, etc.) from resumes in PDF format.
- **Job Description Embedding**: Converts job descriptions into embeddings for similarity analysis.
- **Resume Matching**: Identifies and ranks resumes based on their compatibility with job descriptions.
- **Interactive User Interface**: Allows users to upload resumes and input job descriptions via a Streamlit web app.

## Deployment
The application is live and accessible at:  
[https://matchcv.streamlit.app/]

## Installation and Setup
To run the project locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- Necessary API keys for Hugging Face and Groq (set in a `.env` file)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/aliarmaghan/Cogniquest-AI-Assissment.git
   cd <Resume_Matching>
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required Python libraries:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set API keys**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   HF_TOKEN=your_huggingface_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:  
   Open [http://localhost:8501](http://localhost:8501) in your web browser.

## Project Structure
```
project-name/
│
├── app.py                  # Main application file
├── embedding.py            # Handles embeddings and similarity search
├── parsing.py              # PDF parsing and summarization
├── requirements.txt        # List of dependencies
├── README.md               # Project overview and setup instructions
├── temp/                   # Temporary files directory (add to .gitignore)
└── .gitignore              # File to exclude unnecessary files
```

## Methodology
1. **Resume Parsing**:
   - Extracts text content from PDF files using `pdfplumber` and Docling libraries.
   - Summarizes the resumes using Groq API with a custom prompt.

2. **Embedding Creation**:
   - Converts resume text and job descriptions into embeddings using the Hugging Face `all-MiniLM-L6-v2` model.
   - Stores embeddings in a FAISS vector database.

3. **Similarity Search**:
   - Matches job description embeddings with stored resume embeddings using cosine similarity.
   - Ranks resumes based on their similarity scores.

## Visualizations
- Similarity score distribution between resumes and job descriptions.
- Examples of matched resume segments with job descriptions.
- Optional: Embedding space visualization using t-SNE or PCA.

## Future Enhancements
- Add support for multiple file formats beyond PDFs.
- Implement more advanced explainability features to highlight specific matching sections.
- Optimize embeddings using fine-tuned models for domain-specific needs.

## Acknowledgments
- **Streamlit** for building the interactive UI.
- **Hugging Face** for providing embedding models.
- **Groq API** for advanced summarization capabilities.

## License
This project is licensed under the [MIT License](LICENSE).
