
"""LLM agents, intent routing, prompt templates and LCEL chains."""

import json
import re
import time
import random
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_BASE = """You are Paper-to-Presentation & Viva Agent, an academic assistant.

Use only supplied research-paper context for paper-specific facts.
Never invent authors, datasets, metrics, citations, experiments or results.

If the supplied context does not contain an answer, say that the information
is not available in the retrieved passages.

Use clear, student-friendly academic English.
Do not reveal private chain-of-thought.
Give concise conclusions and brief reasons.
"""


# --------------------------------------------------
# 1. BUILD GEMINI MODEL
# --------------------------------------------------

def build_llm(model_name: str = "gemini-3.8-flash"):
    """Create the Gemini chat model."""

    import os

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Please add your Gemini API key to the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2,
    )


# --------------------------------------------------
# 2. AUTOMATIC RETRY HANDLING
# --------------------------------------------------

def invoke_with_retry(chain, inputs: Dict[str, Any], retries: int = 4):
    """
    Invoke a LangChain chain and retry temporary Gemini errors.

    Retry delays:
    Attempt 1 -> 2 seconds
    Attempt 2 -> 4 seconds
    Attempt 3 -> 8 seconds

    Invalid API keys and other permanent errors are not retried.
    """

    transient_markers = (
        "503",
        "UNAVAILABLE",
        "high demand",
        "temporarily unavailable",
        "502",
        "504",
        "RESOURCE_EXHAUSTED",
        "429",
        "rate limit",
        "overloaded",
    )

    for attempt in range(retries):

        try:
            return chain.invoke(inputs)

        except Exception as exc:

            message = str(exc)

            is_transient = any(
                marker.lower() in message.lower()
                for marker in transient_markers
            )

            # Do not retry permanent errors.
            if not is_transient:
                raise

            # All retry attempts exhausted.
            if attempt == retries - 1:

                raise RuntimeError(
                    "Gemini is currently unavailable or overloaded "
                    "after several attempts.\n\n"
                    "Please wait a minute and try again. "
                    "Your uploaded PDF does not need to be uploaded "
                    "again if it is still visible in the app.\n\n"
                    f"Original error: {message}"
                ) from exc

            # Exponential backoff.
            delay = (2 ** (attempt + 1)) + random.uniform(0, 0.5)

            print(
                f"Gemini temporarily unavailable. "
                f"Retry {attempt + 1}/{retries - 1} "
                f"in {delay:.1f} seconds..."
            )

            time.sleep(delay)


# --------------------------------------------------
# 3. CREATE LANGCHAIN CHAIN
# --------------------------------------------------

def make_chain(llm, system_prompt: str, user_prompt: str):
    """Connect prompt -> Gemini model -> string parser."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


# --------------------------------------------------
# 4. EXTRACT JSON FROM MODEL RESPONSE
# --------------------------------------------------

def _extract_json(raw: str) -> Any:
    """Extract JSON even if the model adds markdown formatting."""

    text = raw.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


# --------------------------------------------------
# 5. KEYWORD-BASED INTENT ROUTING FALLBACK
# --------------------------------------------------

def _keyword_route(request: str) -> Dict[str, Any]:

    q = request.lower()

    if any(w in q for w in ["slide", "ppt", "presentation"]):
        intent = "slides"

    elif any(w in q for w in ["viva", "examiner", "oral question"]):
        intent = "viva"

    elif any(
        w in q
        for w in [
            "concept",
            "contribution",
            "method",
            "result",
            "extract"
        ]
    ):
        intent = "concepts"

    elif any(w in q for w in ["summar", "abstract", "overview"]):
        intent = "summary"

    elif any(
        w in q
        for w in ["explain", "what is", "how does", "why did"]
    ):
        intent = "explain"

    else:
        intent = "chat"

    return {
        "intent": intent,
        "needs_retrieval": intent != "chat",
        "retrieval_query": request,
        "reason": (
            "Keyword fallback used because the model router "
            "did not return valid JSON."
        ),
    }


# --------------------------------------------------
# 6. INTENT DETECTION AGENT
# --------------------------------------------------

def detect_intent(llm, user_request: str) -> Dict[str, Any]:
    """Classify the user's request."""

    system = """
    Classify the student's request.

    Return ONLY valid JSON with this structure:

    {
        "intent": "summary|slides|concepts|viva|explain|chat",
        "needs_retrieval": true,
        "retrieval_query": "short search query",
        "reason": "one short sentence"
    }

    Use needs_retrieval=false for greetings and general questions
    unrelated to the uploaded research paper.
    """

    try:

        chain = make_chain(
            llm,
            system,
            "{request}"
        )

        raw = invoke_with_retry(
            chain,
            {"request": user_request}
        )

        result = _extract_json(raw)

        allowed = {
            "summary",
            "slides",
            "concepts",
            "viva",
            "explain",
            "chat"
        }

        if result.get("intent") not in allowed:
            raise ValueError("Router returned an unsupported intent.")

        result["needs_retrieval"] = bool(
            result.get("needs_retrieval", True)
        )

        result["retrieval_query"] = str(
            result.get("retrieval_query") or user_request
        )

        result["reason"] = str(
            result.get("reason") or "Classified by the intent router."
        )

        return result

    except Exception:
        return _keyword_route(user_request)


# --------------------------------------------------
# 7. SUMMARIZER AGENT
# --------------------------------------------------

def summarize(llm, context: str) -> str:
    """Generate a structured research-paper summary."""

    system = SYSTEM_BASE + """
    You are the Summarizer Agent.
    """

    user = """
    Create a structured research-paper summary with these headings:

    1. Paper overview and research problem
    2. Objectives
    3. Methodology or system architecture
    4. Dataset and experimental setup (only if stated)
    5. Results and findings (do not invent values)
    6. Main contributions
    7. Limitations and future scope

    Use concise paragraphs and bullets.
    Include source page numbers when they appear.

    If a requested detail is absent, write:
    'Not stated in the retrieved paper passages'.

    PAPER PASSAGES:

    {context}
    """

    chain = make_chain(
        llm,
        system,
        user
    )

    return invoke_with_retry(
        chain,
        {"context": context}
    )


# --------------------------------------------------
# 8. CONCEPT EXTRACTOR AGENT
# --------------------------------------------------

def extract_concepts(llm, context: str) -> str:
    """Extract important research-paper concepts."""

    system = SYSTEM_BASE + """
    You are the Concept-Extractor Agent.
    """

    user = """
    Extract the following from the supplied research-paper passages:

    - Research problem and motivation
    - Research gap
    - Main contributions and novelty claims
    - Methods, algorithms and architecture
    - Important technical terms with simple definitions
    - Dataset, experimental setup and evaluation metrics
    - Results and interpretation
    - Limitations and future work

    Use clear headings and bullets.
    Do not infer facts that are not present.

    PAPER PASSAGES:

    {context}
    """

    chain = make_chain(
        llm,
        system,
        user
    )

    return invoke_with_retry(
        chain,
        {"context": context}
    )


# --------------------------------------------------
# 9. CONCEPT EXPLANATION AGENT
# --------------------------------------------------

def explain_concept(
    llm,
    context: str,
    question: str
) -> str:
    """Explain a concept using simple language."""

    system = SYSTEM_BASE + """
    You are a patient university tutor.
    """

    user = """
    Answer the student's question in simple language.

    First explain the concept generally.
    Then connect it to the research paper using only the supplied passages.

    If the paper does not discuss the concept, say so clearly.

    PAPER PASSAGES:

    {context}

    STUDENT QUESTION:

    {question}
    """

    chain = make_chain(
        llm,
        system,
        user
    )

    return invoke_with_retry(
        chain,
        {
            "context": context,
            "question": question
        }
    )


# --------------------------------------------------
# 10. VIVA GENERATION AGENT
# --------------------------------------------------

def generate_viva(
    llm,
    context: str,
    level: str = "Mixed"
) -> str:
    """Generate viva questions and model answers."""

    system = SYSTEM_BASE + """
    You are the Viva Agent preparing a student for an examiner.
    """

    user = """
    Generate 12 likely viva questions with concise model answers.

    Difficulty: {level}

    Include questions about:

    - Understanding of the research paper
    - Research motivation
    - Problem statement
    - Methodology
    - Algorithms
    - System design
    - Results
    - Limitations
    - Comparisons
    - Future work

    Include two challenging follow-up questions.

    Answers should sound natural when spoken aloud.
    Do not invent facts.

    PAPER PASSAGES:

    {context}
    """

    chain = make_chain(
        llm,
        system,
        user
    )

    return invoke_with_retry(
        chain,
        {
            "context": context,
            "level": level
        }
    )


# --------------------------------------------------
# 11. POWERPOINT SLIDE GENERATION AGENT
# --------------------------------------------------

def generate_slides(
    llm,
    context: str,
    slide_count: int = 8
) -> List[Dict[str, Any]]:
    """Generate structured PowerPoint slide content."""

    system = SYSTEM_BASE + """
    You are the Slide-Generator Agent.
    """

    user = """
    Create exactly {count} content slides from the research paper.

    Return ONLY valid JSON in this exact schema:

    {{
        "slides": [
            {{
                "title": "Slide title",
                "bullets": [
                    "point 1",
                    "point 2",
                    "point 3"
                ],
                "speaker_notes": "Short explanation"
            }}
        ]
    }}

    Suggested presentation flow:

    1. Title and introduction
    2. Background and problem statement
    3. Research objectives
    4. Research gap
    5. Methodology
    6. System architecture or workflow
    7. Experiments and results
    8. Conclusion and future work

    Use 3 to 5 concise bullets per slide.
    Do not invent missing information.

    PAPER PASSAGES:

    {context}
    """

    chain = make_chain(
        llm,
        system,
        user
    )

    raw = invoke_with_retry(
        chain,
        {
            "context": context,
            "count": slide_count
        }
    )

    try:

        data = _extract_json(raw)

        slides = data.get("slides", [])

        clean = []

        for slide in slides[:slide_count]:

            bullets = slide.get("bullets", [])

            if isinstance(bullets, str):
                bullets = [bullets]

            clean.append({
                "title": str(
                    slide.get("title", "Key points")
                ),

                "bullets": [
                    str(x) for x in bullets[:6]
                ],

                "speaker_notes": str(
                    slide.get("speaker_notes", "")
                ),
            })

        if not clean:
            raise ValueError(
                "No slide records were returned."
            )

        return clean

    except Exception:

        lines = [
            line.strip(" -*•\t")
            for line in raw.splitlines()
            if line.strip()
        ]

        return [
            {
                "title": "Research Paper Overview",

                "bullets": (
                    lines[:8]
                    or ["Review the uploaded research paper."]
                ),

                "speaker_notes": (
                    "This slide was created using a text fallback "
                    "because the model response was not valid JSON."
                ),
            }
        ]
