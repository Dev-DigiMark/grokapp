import os, re
import base64
import openai 
import pytesseract
import pandas as pd
from fpdf import FPDF
import streamlit as st
from io import BytesIO
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fpdf.enums import XPos, YPos
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any

load_dotenv()

XAI_API_KEY = os.environ.get("GROK_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "legal-documents")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# === Pinecone Setup ===
def initialize_pinecone():
    """Initialize Pinecone client and index"""
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = os.environ.get("PINECONE_INDEX_NAME", "legal-documents")
        dimension = 384  # or 1536 if using OpenAI embeddings

        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',  # or 'gcp'
                    region='us-west-2'  # or your region
                )
            )

        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Pinecone initialization error: {str(e)}")
        return None

# === Document Processing for RAG ===
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def process_document_for_rag(file_obj, file_name: str, deal_id: str) -> List[Dict[str, Any]]:
    """Process document and create chunks for vector storage"""
    chunks = []
    
    try:
        if file_obj.type == "application/pdf":
            # Process PDF
            pdf_reader = PdfReader(file_obj)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
        elif file_obj.type.lower() in ["image/jpeg", "image/jpg", "image/png"]:
            # Process image using OpenAI Vision
            text = process_image_with_openai_classification_and_action(file_obj)
            if isinstance(text, dict):
                text = text.get("content", str(text))
        
        else:
            return chunks
        
        # Clean and chunk the text
        cleaned_text = clean_text(text.strip())
        text_chunks = chunk_text(cleaned_text)
        
        # Create chunk documents
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{deal_id}_{file_name}_{i}_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "deal_id": deal_id,
                    "file_name": file_name,
                    "file_type": file_obj.type,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            })
        
    except Exception as e:
        st.error(f"Error processing document {file_name}: {str(e)}")
    
    return chunks

# === Vector Database Operations ===
def store_documents_in_pinecone(chunks: List[Dict[str, Any]], index) -> bool:
    """Store document chunks in Pinecone"""
    try:
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk in chunks:
            # Generate embedding
            embedding = model.encode(chunk["text"]).tolist()
            
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": chunk["metadata"]
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        return True
        
    except Exception as e:
        st.error(f"Error storing documents in Pinecone: {str(e)}")
        return False

def retrieve_relevant_chunks(query: str, deal_id: str, index, top_k: int = 5) -> List[str]:
    """Retrieve relevant document chunks for a query"""
    try:
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            filter={"deal_id": deal_id},
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract relevant chunks
        relevant_chunks = []
        for match in results.matches:
            if match.metadata and "text" in match.metadata:
                relevant_chunks.append(match.metadata["text"])
        
        return relevant_chunks
        
    except Exception as e:
        st.error(f"Error retrieving chunks from Pinecone: {str(e)}")
        return []

# === Enhanced Grok Logic with RAG ===
def generate_report_with_grok_rag(deal_data, index):
    """Generate report using Grok with RAG-enhanced context"""
    
    # Create base prompt
    base_prompt = load_prompt_template().format(deal_data=deal_data)
    
    # Get relevant chunks from vector database
    query = f"legal funding risk assessment for {deal_data.get('deal_id', 'lead')}"
    relevant_chunks = retrieve_relevant_chunks(query, deal_data.get('deal_id', ''), index)
    
    # Build enhanced prompt with RAG context
    enhanced_prompt = base_prompt
    
    if relevant_chunks:
        enhanced_prompt += "\n\n=== RELEVANT DOCUMENT CONTEXT ===\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            enhanced_prompt += f"Document Chunk {i}: {chunk}\n\n"
        enhanced_prompt += "=== END DOCUMENT CONTEXT ===\n\n"
        enhanced_prompt += "Please use the above document context to enhance your analysis and provide more detailed insights in the report."
    
    # Add uploaded files summary
    uploaded_reports = deal_data.get("uploaded_reports", [])
    if uploaded_reports:
        enhanced_prompt += "\n\nAttached files for this lead (with extracted summaries):\n"
        for f in uploaded_reports:
            enhanced_prompt += f"- {f['name']} (type: {f['type']})\n"
            if f.get('summary'):
                enhanced_prompt += f"  Extracted summary: {f['summary']}\n"
    
    # Generate response with Grok
    response = grok.chat_completion(
        model="grok-3",
        messages=[{"role": "user", "content": enhanced_prompt}],
        temperature=0.3,
        max_tokens=10000
    )
    
    return response

# === GrokClient ===
class GrokClient:
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=1000):
        completion = client.chat.completions.create(
            model="grok-3",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

grok = GrokClient()

def load_prompt_template():
    with open("prompt.txt", "r") as f:
        return f.read()

def save_prompt_template(content):
    with open("prompt.txt", "w") as f:
        f.write(content)

# === Read Users ===
def load_users():
    users = {}
    with open("users.txt", "r") as file:
        for line in file:
            if ":" in line:
                user, pwd = line.strip().split(":", 1)
                users[user] = pwd
    return users

# === Authentication ===
def login_screen(users):
    st.title("🔐 Login to Access the Report Generator")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.is_admin = (username == "admin")  # Set admin status
            st.rerun()
        else:
            st.error("Invalid username or password.")

# === Admin Screen ===
def admin_screen():
    st.title("👨‍💼 Admin Dashboard - Prompt Template Editor")
    
    # Load current prompt template
    current_prompt = load_prompt_template()
    
    # Create a text area for editing
    edited_prompt = st.text_area(
        "Edit Prompt Template",
        value=current_prompt,
        height=500,
        key="prompt_editor"
    )
    
    # Save button
    if st.button("Save Changes"):
        try:
            save_prompt_template(edited_prompt)
            st.success("Prompt template updated successfully!")
        except Exception as e:
            st.error(f"Error saving prompt template: {str(e)}")
    
    # Add a button to switch to main app
    if st.button("Switch to Report Generator"):
        st.session_state.show_admin = False
        st.rerun()

# === Grok Logic ===
def generate_report_with_grok(deal_data):
    prompt = load_prompt_template().format(deal_data=deal_data)
    # If there are uploaded files, append a summary to the prompt
    uploaded_reports = deal_data.get("uploaded_reports", [])
    if uploaded_reports:
        prompt += "\n\nAttached files for this lead (with extracted summaries):\n"
        for f in uploaded_reports:
            prompt += f"- {f['name']} (type: {f['type']})\n"
            if f.get('summary'):
                prompt += f"  Extracted summary: {f['summary']}\n"
    response = grok.chat_completion(
        model="grok-3",  # Updated to grok-3 for consistency
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=10000
    )
    return response

def clean_text(text):
    replacements = {
        '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf')
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf')
        self.set_font('DejaVu', '', 12)

# Utility to replace non-ASCII characters with '-'
def safe_ascii(text):
    return ''.join(c if ord(c) < 128 else '-' for c in text)

def parse_markdown_line(pdf, line):
    line = safe_ascii(line)
    # HEADERS
    if line.startswith("### "):
        pdf.set_font('DejaVu', 'B', 12)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, line[4:])
        pdf.ln(1)
    elif line.startswith("## "):
        pdf.set_font('DejaVu', 'B', 14)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, line[3:])
        pdf.ln(2)
    elif line.startswith("# "):
        pdf.set_font('DejaVu', 'B', 16)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 10, line[2:])
        pdf.ln(4)
    # BULLETS
    elif line.strip().startswith("- "):
        pdf.set_font('DejaVu', '', 12)
        bullet_text = line.strip()[2:].strip()
        try:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 8, f"- {bullet_text}")
        except Exception as e:
            print(f"Error with line: {repr(bullet_text)}")
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 8, f"- [UNRENDERABLE]")
    # BOLD TEXT
    elif "**" in line:
        pdf.set_font('DejaVu', '', 12)
        parts = re.split(r'(\*\*.*?\*\*)', line)
        pdf.set_x(pdf.l_margin)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                bold_text = part[2:-2]
                pdf.set_font('DejaVu', 'B', 12)
                pdf.write(8, bold_text)
                pdf.set_font('DejaVu', '', 12)
            else:
                pdf.write(8, part)
        pdf.ln(8)
    else:
        pdf.set_font('DejaVu', '', 12)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, line)
        pdf.ln(1)

def create_pdf(person_name, report):
    pdf = PDF()
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'Case ID – Decision', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)
    lines = report.split("\n")
    for line in lines:
        if line.strip():
            safe_line = line.encode('utf-8', 'replace').decode()
            parse_markdown_line(pdf, safe_line)
    return BytesIO(pdf.output(dest='S'))


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Make sure to set this in your .env file
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# === OpenAI Image Processing ===
def process_image_with_openai_classification_and_action(image_file):
    try:
        # Read and encode
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Ask GPT to classify the image
        classify_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 
                            "Look at this image and tell me clearly what type of image this is. "
                            "Is it a document, ID card, form, receipt, or is it a photo of a scene, accident, street, or something else? "
                            "Give a short label like 'Driver License', 'Accident Scene', 'Receipt', etc."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        image_type = classify_response.choices[0].message.content.strip()
        
        # Now decide how to process based on type
        if any(keyword in image_type.lower() for keyword in ["license", "id", "document", "form", "card"]):
            prompt_text = (
                "Extract all text exactly as it appears from this image, including names, numbers, dates, and personal info. "
                "Return only the text."
            )
        else:
            prompt_text = (
                "Describe in detail what is visible in this image. "
                "Focus on objects, people, vehicles, and the situation."
            )
        
        # Final processing
        final_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        content = final_response.choices[0].message.content.strip()
        
        # If OpenAI refuses, fallback to Tesseract OCR
        if (
            any(keyword in image_type.lower() for keyword in ["license", "id", "document", "form", "card"])
            and ("can't assist" in content.lower() or "cannot assist" in content.lower())
        ):
            # Use Tesseract OCR
            import io
            from PIL import Image
            image_file.seek(0)  # Reset file pointer
            image = Image.open(io.BytesIO(image_file.read()))
            ocr_text = pytesseract.image_to_string(image)
            content = ocr_text.strip()

        return content
        

    except Exception as e:
        return {"type": "Error", "content": f"[OpenAI Vision Error: {str(e)}]"}

    
# === Main App ===
def main_app():
    if "uploader_reset" not in st.session_state:
        st.session_state["uploader_reset"] = 0
    # Add a button for admin to switch to admin screen
    if st.session_state.get("is_admin", False):
        if st.sidebar.button("Switch to Admin Dashboard"):
            st.session_state.show_admin = True
            st.rerun()
    
    st.title("📑 Legal Funding Risk Report Generator (Grok)")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key=f"csv_uploader_{st.session_state['uploader_reset']}"
    )
    
    if uploaded_file:
        # Extract deal id from the uploaded file name
        deal_id = uploaded_file.name.split("-")[0]
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(df)} records.")
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
        if "report_buffers" not in st.session_state:
            st.session_state.report_buffers = {}
        for idx, row in df.iterrows():
            # Use deal_id as the person_name for file naming
            person_name = deal_id
            deal_data = row.to_dict()
            st.write(f"---\n### Lead: **{person_name}**")
            file_key = f"files_{idx}_{st.session_state['uploader_reset']}"
            uploaded_files = st.file_uploader(
                f"Upload Police/Accident Reports for {person_name} (PDF/JPG/PNG)",
                type=["pdf", "jpg", "jpeg", "png", "JPEG", "JPG", "PNG"],
                key=file_key,
                accept_multiple_files=True
            )
            # Store in session and process for RAG
            if uploaded_files:
                st.session_state.uploaded_files[person_name] = uploaded_files
                file_info_list = []
                
                # Initialize Pinecone for RAG
                pinecone_index = initialize_pinecone()
                
                for f in uploaded_files:
                    file_summary = ""
                    # Extract text summary from PDF
                    if f.type == "application/pdf":
                        try:
                            pdf_reader = PdfReader(f)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() or ""
                            file_summary = clean_text(text.strip().replace("\n", " ")[:500])
                            
                            # Process for RAG if Pinecone is available
                            if pinecone_index:
                                chunks = process_document_for_rag(f, f.name, person_name)
                                if chunks:
                                    store_documents_in_pinecone(chunks, pinecone_index)
                                    st.success(f"✅ Stored {len(chunks)} chunks from {f.name} in vector database")
                                    
                        except Exception as e:
                            file_summary = f"[Could not extract PDF text: {str(e)}]"
                            
                    # Extract text summary from images with enhanced preprocessing
                    elif f.type.lower() in ["image/jpeg", "image/jpg", "image/png"]:
                        try:
                            # Use OpenAI for image text extraction
                            file_summary = process_image_with_openai_classification_and_action(f)
                            
                            # Process for RAG if Pinecone is available
                            if pinecone_index:
                                chunks = process_document_for_rag(f, f.name, person_name)
                                if chunks:
                                    store_documents_in_pinecone(chunks, pinecone_index)
                                    st.success(f"✅ Stored {len(chunks)} chunks from {f.name} in vector database")
                                    
                        except Exception as e:
                            file_summary = f"[Image Extraction Error: {str(e)}]"
                    
                    file_info_list.append({
                        "name": f.name,
                        "type": f.type,
                        "summary": file_summary
                    })
                deal_data["uploaded_reports"] = file_info_list
            else:
                deal_data["uploaded_reports"] = []
            # Generate Report Button
            if st.button(f"Generate Report for {person_name}", key=f"gen_{idx}"):
                try:
                    st.write("Generating report...")
                    # Initialize Pinecone index
                    pinecone_index = initialize_pinecone()
                    if pinecone_index:
                        report = generate_report_with_grok_rag(deal_data, pinecone_index)
                    else:
                        report = generate_report_with_grok(deal_data) # Fallback to original if Pinecone fails
                    pdf_buffer = create_pdf(person_name, report)
                    st.session_state.report_buffers[person_name] = pdf_buffer
                    st.success("Report generated!")
                except Exception as e:
                    st.error(f"Error processing {person_name}: {str(e)}")
                    st.exception(e)
            # Show download button if report is generated
            if person_name in st.session_state.get("report_buffers", {}):
                st.download_button(
                    label=f"Download Report for {person_name}",
                    data=st.session_state.report_buffers[person_name],
                    file_name=f"{person_name}.pdf",
                    mime="application/pdf"
                )
                # Add a reload button
                if st.button(f"Reload Screen for {person_name}", key=f"reload_{idx}"):
                    # Clear all session state except authentication and admin info
                    keys_to_keep = {"authenticated", "username", "is_admin", "show_admin"}
                    keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
                    for k in keys_to_delete:
                        del st.session_state[k]
                    # Increment uploader_reset to force file_uploader widgets to reset
                    st.session_state["uploader_reset"] = st.session_state.get("uploader_reset", 0) + 1
                    st.rerun()

# === App Entry Point ===
def main():
    users = load_users()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False

    if not st.session_state.authenticated:
        login_screen(users)
    else:
        if st.session_state.get("is_admin", False) and st.session_state.show_admin:
            admin_screen()
        else:
            main_app()

if __name__ == "__main__":
    main()
