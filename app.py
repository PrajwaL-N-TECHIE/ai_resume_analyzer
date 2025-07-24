import streamlit as st
import requests
import json
from PyPDF2 import PdfReader
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .st-emotion-cache-z5fcl4 { /* This targets the container for a card-like effect */
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .stTextArea textarea {
        height: 300px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_txt(file_bytes):
    """Extracts text from a TXT file."""
    try:
        return file_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

# --- AI Interaction Function ---
@st.cache_data(show_spinner="🤖 AI is analyzing your resume...")
def get_ai_analysis(resume_text, job_description, api_key):
    """
    Sends resume and job description to Gemini API for analysis.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    # A structured schema for the AI's response
    analysis_schema = {
        "type": "OBJECT",
        "properties": {
            "match_score": {"type": "INTEGER", "description": "A score from 0 to 100 representing how well the resume matches the job description."},
            "summary": {"type": "STRING", "description": "A brief, professional summary of the analysis."},
            "strengths": {"type": "STRING", "description": "What aspects of the resume are a strong match for the job."},
            "improvements": {"type": "STRING", "description": "Actionable suggestions for improving the resume for this specific job."},
            "keywords_to_add": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "A list of important keywords from the job description that are missing from the resume."}
        },
        "required": ["match_score", "summary", "strengths", "improvements", "keywords_to_add"]
    }

    prompt = f"""
    You are an expert career coach and resume analyzer. Your task is to analyze the provided resume against the given job description and provide a detailed, constructive analysis.

    **Resume Text:**
    ---
    {resume_text}
    ---

    **Job Description:**
    ---
    {job_description}
    ---

    Please provide your analysis in the structured JSON format requested. The match score should be a realistic assessment of the candidate's fit based ONLY on the provided documents.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": analysis_schema
        }
    }

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if "candidates" in result:
            return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
        else:
            return {"error": "The AI returned an unexpected response."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# --- Main Application UI ---
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("Get instant feedback on how well your resume matches a job description.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google AI API Key", type="password")
    uploaded_resume = st.file_uploader("Upload Your Resume", type=["pdf", "txt"])
    job_description = st.text_area("Paste the Job Description Here")
    analyze_button = st.button("Analyze Resume", use_container_width=True)

# --- Main Content Area ---
if analyze_button:
    if not api_key:
        st.error("Please enter your Google AI API Key in the sidebar.")
    elif not uploaded_resume:
        st.error("Please upload your resume.")
    elif not job_description:
        st.error("Please paste the job description.")
    else:
        # Process the file
        resume_bytes = uploaded_resume.getvalue()
        if uploaded_resume.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_bytes)
        else:
            resume_text = extract_text_from_txt(resume_bytes)

        if resume_text:
            analysis_result = get_ai_analysis(resume_text, job_description, api_key)

            if "error" in analysis_result:
                st.error(analysis_result["error"])
            else:
                st.header("Analysis Results")

                # Display Match Score
                st.subheader("Match Score")
                score = analysis_result.get('match_score', 0)
                st.metric(label="Your resume's match for this role is:", value=f"{score}%")
                st.progress(score)
                st.markdown(f"**Summary:** {analysis_result.get('summary', 'N/A')}")
                st.markdown("---")

                # Display Strengths and Improvements
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Strengths")
                    st.markdown(analysis_result.get('strengths', 'N/A').replace('\n', '\n\n'))
                with col2:
                    st.subheader("💡 Improvements")
                    st.markdown(analysis_result.get('improvements', 'N/A').replace('\n', '\n\n'))
                st.markdown("---")

                # Display Missing Keywords
                st.subheader("🔑 Keywords to Add")
                keywords = analysis_result.get('keywords_to_add', [])
                if keywords:
                    st.info("Consider naturally weaving these keywords from the job description into your resume:")
                    # Display keywords in columns for better layout
                    num_columns = 3
                    cols = st.columns(num_columns)
                    for i, keyword in enumerate(keywords):
                        with cols[i % num_columns]:
                            st.markdown(f"- **{keyword}**")
                else:
                    st.success("Great job! It looks like you've covered all the important keywords.")
else:
    st.info("Please provide your API key, resume, and a job description in the sidebar, then click 'Analyze Resume'.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Powered by Streamlit & Google Gemini</p>", unsafe_allow_html=True)