# Legal Funding Risk Report Generator with RAG

A Streamlit web application that generates legal funding risk reports using Grok AI with Retrieval-Augmented Generation (RAG) powered by Pinecone vector database.

## 🚀 Core Features

1. **User Authentication** - Secure login system with admin/user roles
2. **Admin Dashboard** - Edit AI prompt templates for report generation
3. **Lead Data Upload** - Upload CSV files with multiple leads/cases
4. **Document Processing** - Extract text from PDFs and images using OCR and OpenAI Vision
5. **RAG Integration** - Store processed documents in Pinecone vector database for enhanced retrieval
6. **AI-Powered Report Generation** - Generate comprehensive reports using Grok with RAG context
7. **PDF Report Creation** - Download formatted reports as PDFs
8. **Session Management** - Handle multiple leads and file uploads efficiently

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.example.env` to `.env` and fill in your API keys:
```bash
cp .example.env .env
```

Edit `.env` with your API keys:
```
GROK_API_KEY = "your_xai_api_key"
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX_NAME = "legal-documents"
```

### 3. Set Up User Accounts
Edit `users.txt` to add usernames and passwords:
```
admin:adminpassword
user1:user1password
```

### 4. Run the Application
```bash
streamlit run main.py
```

## 🔧 RAG Implementation Details

### Document Processing Flow:
1. **Upload** - Users upload CSV files and supporting documents
2. **Extract** - Text extraction from PDFs and images using OCR/OpenAI Vision
3. **Chunk** - Split documents into overlapping chunks (500 words, 50 overlap)
4. **Embed** - Generate embeddings using sentence-transformers
5. **Store** - Store vectors in Pinecone with metadata
6. **Retrieve** - Query relevant chunks during report generation
7. **Enhance** - Use retrieved context to improve Grok's report quality

### Vector Database Schema:
- **Index**: `legal-documents`
- **Dimension**: 384 (sentence-transformers/all-MiniLM-L6-v2)
- **Metric**: Cosine similarity
- **Metadata**: deal_id, file_name, file_type, chunk_index, total_chunks

## 📊 Usage Workflow

1. **Login** with credentials from `users.txt`
2. **Upload CSV** containing lead data
3. **Attach Documents** for each lead (PDFs, images)
4. **Process Documents** - Automatic RAG processing and storage
5. **Generate Reports** - AI-powered reports with RAG-enhanced context
6. **Download PDFs** - Formatted reports for each lead

## 🔍 RAG Benefits

- **Enhanced Context**: Reports include relevant document snippets
- **Better Accuracy**: AI has access to specific case details
- **Comprehensive Analysis**: Combines lead data with document insights
- **Scalable Storage**: Vector database handles large document collections

## 🛡️ Security Features

- User authentication and role-based access
- Secure API key management
- Session state management
- Error handling and fallback mechanisms

## 📝 File Structure

```
grok-repo/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── .env               # Environment variables
├── users.txt          # User credentials
├── prompt.txt         # AI prompt template
├── README.md          # This file
└── fonts/            # PDF fonts
    ├── DejaVuSans.ttf
    └── DejaVuSans-Bold.ttf
```

## 🚨 Troubleshooting

### Common Issues:
1. **Pinecone Connection**: Ensure API key and environment are correct
2. **Tesseract OCR**: Install Tesseract on your system
3. **OpenAI Vision**: Verify OpenAI API key has vision access
4. **Memory Issues**: Large documents may require chunking optimization

### Fallback Mechanisms:
- If Pinecone fails, falls back to original Grok generation
- If image processing fails, uses Tesseract OCR as backup
- Graceful error handling for all external API calls