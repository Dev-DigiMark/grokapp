import streamlit as st
import pandas as pd
from fpdf import FPDF
import os
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from fpdf.enums import XPos, YPos
import re
import base64

load_dotenv()

XAI_API_KEY = os.environ.get("GROK_API_KEY")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

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
        prompt += "\n\nAttached files for this lead:\n"
        for f in uploaded_reports:
            prompt += f"- {f['name']} (type: {f['type']})\n"
    response = grok.chat_completion(
        model="grok-1-chat",
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

# === Main App ===
def main_app():
    # Add a button for admin to switch to admin screen
    if st.session_state.get("is_admin", False):
        if st.sidebar.button("Switch to Admin Dashboard"):
            st.session_state.show_admin = True
            st.rerun()
    
    st.title("📑 Legal Funding Risk Report Generator (Grok)")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
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
            file_key = f"files_{idx}"
            uploaded_files = st.file_uploader(
                f"Upload Police/Accident Reports for {person_name} (PDF/JPG)",
                type=["pdf", "jpg", "jpeg"],
                key=file_key,
                accept_multiple_files=True
            )
            # Store in session
            if uploaded_files:
                st.session_state.uploaded_files[person_name] = uploaded_files
                file_info_list = []
                deal_data["uploaded_reports"] = file_info_list
            else:
                deal_data["uploaded_reports"] = []
            # Generate Report Button
            if st.button(f"Generate Report for {person_name}", key=f"gen_{idx}"):
                try:
                    report = generate_report_with_grok(deal_data)
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
