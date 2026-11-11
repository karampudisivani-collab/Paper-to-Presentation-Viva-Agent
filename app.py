import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents import (
    build_llm, detect_intent, summarize, extract_concepts,
    explain_concept, generate_viva, generate_slides, make_chain, SYSTEM_BASE
)
from ppt_generator import create_presentation
from rag_engine import ingest_pdf, retrieve_context

load_dotenv()

st.set_page_config(
    page_title="Paper-to-Presentation & Viva Agent",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2rem;font-weight:800;color:#24577f;margin-bottom:.2rem;}
    .sub-title {color:#64748b;margin-bottom:1.2rem;}
    div[data-testid="stMetric"] {background:#f1f5f9;padding:12px;border-radius:10px;}
    </style>
    <div class="main-title">🎓 Paper-to-Presentation &amp; Viva Agent</div>
    <div class="sub-title">Upload a research paper, understand it, create slides and prepare for your viva.</div>
    """,
    unsafe_allow_html=True,
)

# Session state
defaults = {
    "vectorstore": None,
    "paper_name": None,
    "paper_stats": (0, 0),
    "chat_history": [],
    "summary_result": None,
    "concepts_result": None,
    "viva_result": None,
    "slides_result": None,
    "pptx_bytes": None,
    "last_route": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar: key and PDF upload
with st.sidebar:
    st.header("1. Configure Gemini")
    env_key = os.getenv("GOOGLE_API_KEY", "")
    entered_key = st.text_input(
        "Google Gemini API key",
        value="",
        type="password",
        help="Paste your key here, or place it in the .env file. It is not displayed.",
    )
    api_key = entered_key.strip() or env_key.strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key detected")
    else:
        st.warning("Add your Gemini API key to generate summaries and answers.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    st.divider()
    st.header("2. Upload research paper")
    uploaded = st.file_uploader("Choose a research paper (PDF)", type=["pdf"])

    if st.button("Process PDF", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Choose a PDF first.")
        else:
            with st.spinner("Extracting PDF text, creating local vectors and indexing in ChromaDB..."):
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        temp_path = tmp.name

                    store, page_count, chunk_count = ingest_pdf(
                        temp_path, uploaded.getvalue()
                    )

                    st.session_state.vectorstore = store
                    st.session_state.paper_name = uploaded.name
                    st.session_state.paper_stats = (page_count, chunk_count)
                    st.session_state.chat_history = []
                    st.session_state.summary_result = None
                    st.session_state.concepts_result = None
                    st.session_state.viva_result = None
                    st.session_state.slides_result = None
                    st.session_state.pptx_bytes = None
                    st.success("PDF processed successfully.")
                except Exception as exc:
                    st.error(f"Could not process PDF: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    if st.session_state.vectorstore is not None:
        pages, chunks = st.session_state.paper_stats
        st.success(f"Loaded: {st.session_state.paper_name}")
        st.caption(f"{pages} pages extracted · {chunks} searchable chunks")

    st.divider()
    st.caption("Stack: Streamlit · Python · Gemini 3.8 Flash · LangChain LCEL · ChromaDB")
    st.caption("Document vectors are generated locally to avoid Gemini embedding quotas.")

if st.session_state.vectorstore is None:
    st.info("Add your Gemini API key, upload a PDF in the sidebar, and click Process PDF.")
    st.markdown("### Four agents")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("📝 **Summarizer**\n\nSection-wise paper summary")
    c2.info("💡 **Concept Extractor**\n\nResearch gap, method and findings")
    c3.info("📊 **Slide Generator**\n\nEditable PowerPoint deck")
    c4.info("🎤 **Viva Agent**\n\nExaminer questions and answers")
    st.stop()

if not api_key:
    st.warning("Your PDF is indexed. Add a Gemini API key in the sidebar to use the AI agents.")
    st.stop()

try:
    llm = build_llm(model_name)
except Exception as exc:
    st.error(f"Could not initialize Gemini model '{model_name}': {exc}")
    st.stop()

tab_summary, tab_concepts, tab_slides, tab_viva, tab_chat = st.tabs(
    ["📝 Summary", "💡 Concepts", "📊 Slides", "🎤 Viva Practice", "💬 Ask the Paper"]
)

with tab_summary:
    st.subheader("Summarizer Agent")
    st.write("Creates a structured summary using relevant passages retrieved from your PDF.")
    if st.button("Generate paper summary", key="summary_btn", type="primary"):
        with st.spinner("Summarizing the research paper..."):
            try:
                context = retrieve_context(
                    st.session_state.vectorstore,
                    "abstract research problem objectives methodology experiments results limitations conclusion",
                    k=12,
                )
                st.session_state.summary_result = summarize(llm, context)
                st.session_state.last_route = {
                    "intent": "summary", "needs_retrieval": True,
                    "reason": "Summary task routed to the Summarizer Agent."
                }
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")
    if st.session_state.summary_result:
        st.markdown(st.session_state.summary_result)

with tab_concepts:
    st.subheader("Concept-Extractor Agent")
    st.write("Identifies the research problem, contributions, methods, terminology and results.")
    if st.button("Extract key concepts", key="concept_btn", type="primary"):
        with st.spinner("Extracting concepts from the paper..."):
            try:
                context = retrieve_context(
                    st.session_state.vectorstore,
                    "research gap novel contributions methods algorithm architecture dataset evaluation metrics results",
                    k=12,
                )
                st.session_state.concepts_result = extract_concepts(llm, context)
                st.session_state.last_route = {
                    "intent": "concepts", "needs_retrieval": True,
                    "reason": "Concept task routed to the Concept-Extractor Agent."
                }
            except Exception as exc:
                st.error(f"Concept extraction failed: {exc}")
    if st.session_state.concepts_result:
        st.markdown(st.session_state.concepts_result)

with tab_slides:
    st.subheader("Slide-Generator Agent")
    slide_count = st.slider("Number of content slides", 5, 12, 8)
    st.write("Creates presentation-ready slide content and an editable PowerPoint file.")
    if st.button("Generate PowerPoint", key="slides_btn", type="primary"):
        with st.spinner("Generating your presentation..."):
            try:
                context = retrieve_context(
                    st.session_state.vectorstore,
                    "paper title abstract introduction problem objectives methodology system architecture experiments results conclusion",
                    k=14,
                )
                slides = generate_slides(llm, context, slide_count)
                title = Path(st.session_state.paper_name).stem
                st.session_state.pptx_bytes = create_presentation(
                    slides, paper_title=title
                )
                st.session_state.slides_result = slides
                st.session_state.last_route = {
                    "intent": "slides", "needs_retrieval": True,
                    "reason": "Presentation task routed to the Slide-Generator Agent."
                }
            except Exception as exc:
                st.error(f"PowerPoint generation failed: {exc}")

    if st.session_state.slides_result:
        for idx, slide in enumerate(st.session_state.slides_result, 1):
            with st.expander(f"Slide {idx}: {slide.get('title', 'Key points')}", expanded=(idx == 1)):
                for bullet in slide.get("bullets", []):
                    st.markdown(f"- {bullet}")
                if slide.get("speaker_notes"):
                    st.caption(f"Speaker notes: {slide['speaker_notes']}")
        if st.session_state.pptx_bytes:
            safe_name = Path(st.session_state.paper_name).stem.replace(" ", "_")
            st.download_button(
                "⬇️ Download editable PowerPoint (.pptx)",
                data=st.session_state.pptx_bytes,
                file_name=f"{safe_name}_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

with tab_viva:
    st.subheader("Viva Agent")
    level = st.selectbox("Question difficulty", ["Mixed", "Beginner", "Intermediate", "Advanced"])
    if st.button("Generate viva questions", key="viva_btn", type="primary"):
        with st.spinner("Preparing likely examiner questions and model answers..."):
            try:
                context = retrieve_context(
                    st.session_state.vectorstore,
                    "research objective methodology design choices algorithms experiments results limitations future work",
                    k=12,
                )
                st.session_state.viva_result = generate_viva(llm, context, level)
                st.session_state.last_route = {
                    "intent": "viva", "needs_retrieval": True,
                    "reason": "Viva task routed to the Viva Agent."
                }
            except Exception as exc:
                st.error(f"Viva generation failed: {exc}")
    if st.session_state.viva_result:
        st.markdown(st.session_state.viva_result)

with tab_chat:
    st.subheader("Ask questions about your research paper")
    st.caption("The intent router decides whether to retrieve paper passages and which agent should answer.")
    question = st.chat_input("Example: Explain the methodology used in this paper.")
    if question:
        with st.spinner("Detecting intent and preparing an answer..."):
            try:
                route = detect_intent(llm, question)
                context = (
                    retrieve_context(st.session_state.vectorstore, route["retrieval_query"], k=6)
                    if route["needs_retrieval"] else
                    "No paper-specific context was requested."
                )
                intent = route["intent"]
                if intent == "explain":
                    answer = explain_concept(llm, context, question)
                elif intent == "concepts":
                    answer = extract_concepts(llm, context)
                elif intent == "summary":
                    answer = summarize(llm, context)
                elif intent == "viva":
                    answer = generate_viva(llm, context, "Mixed")
                elif intent == "slides":
                    answer = "Open the Slides tab to generate and download an editable PowerPoint presentation."
                else:
                    answer = make_chain(
                        llm,
                        SYSTEM_BASE + "Answer general academic questions briefly. For paper-specific facts, ask the student to ask about the uploaded paper.",
                        "{question}",
                    ).invoke({"question": question})
                st.session_state.chat_history.append({
                    "question": question, "answer": answer, "route": route
                })
                st.session_state.last_route = route
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander("Intent-routing details"):
                st.json(item["route"])

if st.session_state.last_route:
    with st.expander("Latest routing decision"):
        st.json(st.session_state.last_route)
