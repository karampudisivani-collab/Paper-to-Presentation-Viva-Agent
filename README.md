# Paper-to-Presentation & Viva Agent

A Streamlit academic assistant that accepts a research-paper PDF and generates:
- A structured paper summary
- Key concepts, research gap, contributions, methods and findings
- An editable PowerPoint presentation with speaker notes
- Likely viva questions with model answers
- Paper-grounded question answering with intent routing

## Technology
- Python and Streamlit
- Gemini 3.8 Flash through LangChain
- LangChain PromptTemplate and LCEL
- ChromaDB vector database
- Local HashingVectorizer document vectors (no Gemini embedding API calls)
- PyMuPDF and Tesseract OCR
- python-pptx

**Why local vectors?** The earlier version used the Gemini embedding API and hit model deprecation and free-tier rate limits. This version uses local word/bigram vectors for ChromaDB retrieval. Gemini is used only for text generation. This avoids embedding API quota errors and avoids the previous persistent SQLite read-only database issue by using an in-memory Chroma client for each Streamlit session. The local vectorizer is lexical retrieval, not a transformer semantic embedding.

## Requirements
- macOS
- Python 3.11 recommended
- Tesseract OCR (only needed for scanned PDFs)
- A Google Gemini API key

## Setup on Mac / VS Code

### 1. Extract and open
Download the ZIP, extract it, and open the `paper_viva_agent_complete` folder in VS Code.

### 2. Install Python 3.11 if needed
In Terminal:
```bash
brew install python@3.11
```

### 3. Create and activate a virtual environment
From the project folder:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not found, close and reopen VS Code after installing Python, then try again.

### 4. Install packages
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Install OCR engine
```bash
brew install tesseract
```
If you only use selectable-text PDFs, OCR is not needed. The application first tries normal PDF text extraction and only uses OCR for pages with little or no selectable text.

### 6. Configure Gemini API key
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder:
```env
GOOGLE_API_KEY=your_actual_key
GEMINI_MODEL=gemini-3.8-flash
```
Do not share your key or upload `.env` to GitHub.

### 7. Run
```bash
streamlit run app.py
```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How to use
1. Paste your Gemini API key in the sidebar or configure `.env`.
2. Upload a PDF and click **Process PDF**.
3. Open the Summary, Concepts, Slides or Viva tabs.
4. Use Ask the Paper to ask paper-specific questions.
5. Download the generated editable PowerPoint from the Slides tab.

## Seven assignment acceptance criteria

1. **LLM integration:** Gemini 3.8 Flash via `ChatGoogleGenerativeAI`.
2. **Interactive GUI:** Streamlit PDF upload and result tabs.
3. **Prompt engineering:** Separate system/user prompts, task-specific instructions and intent routing.
4. **LangChain core components:** `ChatPromptTemplate`, `StrOutputParser` and LCEL (`prompt | llm | parser`).
5. **Agentic RAG:** Retrieval is selected based on intent and retrieved passages ground paper-specific responses.
6. **Intent detection and routing:** LLM router classifies requests into summary, slides, concepts, viva, explain or chat, with a keyword fallback.
7. **Vector database:** ChromaDB stores locally generated document vectors and performs similarity search.

## Troubleshooting

### `zsh: command not found: pip`
Use:
```bash
python -m pip install -r requirements.txt
```

### `.venv/bin/activate: no such file or directory`
Create it first:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Tesseract not found
```bash
brew install tesseract
```

### Gemini model not found
Check that `.env` contains `GEMINI_MODEL=gemini-3.8-flash`, then restart Streamlit. Model availability can change; use a currently enabled text-generation model in Google AI Studio if needed.

### Gemini quota or billing error
This project does not call Gemini for embeddings, so document indexing does not consume Gemini embedding quota. Summary, concept, slide, viva and chat generation still require Gemini API access and remain subject to your Gemini API quota and billing limits.

### Chroma database readonly error
This version uses an in-memory Chroma client and does not create a local SQLite database folder. Keep the Streamlit process running while using the indexed paper; re-upload and process the PDF after restarting the app.

### PDF text not detected
For scanned PDFs install Tesseract. For very large scanned PDFs, OCR can take several minutes.

## Project files
- `app.py` — Streamlit user interface
- `agents.py` — Gemini model, prompts, LCEL chains, intent router and four agents
- `rag_engine.py` — PDF extraction, OCR, chunking, local vectors, ChromaDB and retrieval
- `ppt_generator.py` — editable PowerPoint generation
- `requirements.txt` — Python dependencies
- `.env.example` — API key and model configuration template
- `sample_docs/` — 15-page selectable-text sample PDF for testing

## Important
This is an academic prototype. Generated content must be checked against the original paper. It should not invent results, citations or author details.
