import json
import os
from io import BytesIO

import streamlit as st
from docx import Document
from pypdf import PdfReader
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages).strip()

    if filename.endswith(".docx"):
        document = Document(BytesIO(file_bytes))
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        ).strip()

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError(
        "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():
    """Create a Gemini client using Streamlit Secrets or an environment variable."""
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_resume(resume_text, job_description):
    """Analyze the resume with Gemini and return structured JSON."""

    prompt = f"""
You are an expert ATS (Applicant Tracking System) resume evaluator,
recruiter, and professional career coach.

Analyze the resume below and provide an ATS-readiness assessment.

IMPORTANT:
1. The ATS score is an estimated ATS-readiness score from 0 to 100.
2. It is NOT an official score from a specific ATS vendor.
3. Never invent experience, employers, education, certifications, projects,
   skills, achievements, dates, or qualifications.
4. Do not recommend keyword stuffing.
5. Only recommend adding a keyword if it is genuinely relevant to the target
   job and the candidate can truthfully claim it.
6. Evaluate readability and ATS parsing.
7. Consider standard section headings.
8. Consider keyword relevance.
9. Consider skills.
10. Consider work experience.
11. Consider education.
12. Consider measurable achievements and action verbs.
13. Consider consistency and professional writing.
14. If no job description is supplied, evaluate general ATS readiness and
    state that keyword matching is limited.

SCORING:
- ats_score: integer from 0 to 100.
- Each score_breakdown value: integer from 0 to 100.
- Do not give 100 unless the resume is exceptionally strong.

Return ONLY valid JSON matching this exact structure:

{{
  "ats_score": 0,
  "score_breakdown": {{
    "formatting": 0,
    "keywords": 0,
    "experience": 0,
    "skills": 0,
    "education": 0,
    "impact": 0
  }},
  "summary": "Short overall assessment.",
  "strengths": [
    "Strength 1",
    "Strength 2",
    "Strength 3"
  ],
  "improvements": [
    {{
      "priority": "High",
      "issue": "Specific issue.",
      "why_it_matters": "Why it matters for ATS or recruiters.",
      "recommendation": "Specific improvement.",
      "example": "A realistic example based only on information already present."
    }}
  ],
  "missing_keywords": [
    "keyword 1"
  ],
  "present_keywords": [
    "keyword 1"
  ],
  "formatting_checks": [
    "Formatting observation."
  ],
  "section_checks": [
    "Section observation."
  ]
}}

TARGET JOB DESCRIPTION:
{job_description.strip() if job_description.strip() else "[No target job description provided]"}

RESUME:
{resume_text[:35000]}
"""

    client = get_gemini_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    response_text = (response.text or "").strip()

    if not response_text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned an invalid JSON response. Please try again."
        ) from exc


# ============================================================
# UI HELPERS
# ============================================================

def get_score(value):
    """Safely convert a score to an integer between 0 and 100."""
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def get_list(value):
    """Return a list for safely rendering AI-generated list fields."""
    return value if isinstance(value, list) else []


def priority_label(priority):
    """Normalize priority text."""
    priority = str(priority or "Medium").strip().lower()

    if priority == "high":
        return "🔴 High"
    if priority == "low":
        return "🟢 Low"

    return "🟡 Medium"


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📄 ATS Resume Analyzer")

st.markdown(
    "Upload your resume to get an **AI-estimated ATS score**, "
    "resume strengths, missing keywords, formatting checks, "
    "and actionable improvement suggestions."
)

st.caption(
    "Supported files: PDF, DOCX, TXT • "
    "For best results, use a text-based PDF rather than a scanned image."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🎯 Target Job")

    job_description = st.text_area(
        "Paste the job description (optional)",
        height=300,
        placeholder=(
            "Paste the complete job description here. "
            "Adding it gives you more useful keyword matching."
        ),
    )

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Upload your resume\n"
        "2. Optionally paste a job description\n"
        "3. Click **Analyze Resume**\n"
        "4. Review your ATS score\n"
        "5. Apply the recommended improvements"
    )

    st.divider()

    st.info(
        "Your Gemini API key is read from Streamlit Secrets "
        "or the GEMINI_API_KEY environment variable. "
        "It is not hard-coded in this application."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Upload a PDF, DOCX, or TXT resume.",
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    try:
        resume_text = extract_text(uploaded_file)

    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

    if not resume_text:
        st.error(
            "No readable text was found in this file. "
            "If this is a scanned PDF, upload a text-based PDF or DOCX."
        )
        st.stop()

    st.success(
        f"Resume loaded successfully: **{uploaded_file.name}**"
    )

    # --------------------------------------------------------
    # RESUME PREVIEW
    # --------------------------------------------------------

    with st.expander("👀 Preview Extracted Resume Text"):
        st.text_area(
            "Extracted text",
            value=resume_text[:15000],
            height=350,
            disabled=True,
            label_visibility="collapsed",
        )

        if len(resume_text) > 15000:
            st.caption(
                "Preview truncated for display. "
                "The analyzer can process additional text."
            )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Gemini is analyzing your resume. Please wait..."
        ):
            try:
                analysis = analyze_resume(
                    resume_text=resume_text,
                    job_description=job_description,
                )

                st.session_state["analysis"] = analysis

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.stop()


# ============================================================
# RESULTS
# ============================================================

analysis = st.session_state.get("analysis")


if analysis:

    st.divider()
    st.header("📊 ATS Analysis Results")

    # --------------------------------------------------------
    # ATS SCORE
    # --------------------------------------------------------

    score = get_score(
        analysis.get("ats_score", 0)
    )

    score_column, summary_column = st.columns(
        [1, 2]
    )

    with score_column:
        st.metric(
            "Estimated ATS Score",
            f"{score}/100",
        )

        st.progress(score / 100)

        if score >= 80:
            st.success("Strong ATS readiness")
        elif score >= 60:
            st.warning("Good foundation, but improvements are recommended")
        else:
            st.error("Several improvements are recommended")

    with summary_column:
        st.subheader("📝 Overall Assessment")
        st.write(
            analysis.get(
                "summary",
                "No summary was returned.",
            )
        )

    # --------------------------------------------------------
    # SCORE BREAKDOWN
    # --------------------------------------------------------

    st.subheader("📈 Score Breakdown")

    breakdown = analysis.get(
        "score_breakdown",
        {},
    )

    categories = [
        ("Formatting", "formatting"),
        ("Keywords", "keywords"),
        ("Experience", "experience"),
        ("Skills", "skills"),
        ("Education", "education"),
        ("Impact", "impact"),
    ]

    columns = st.columns(6)

    for column, (label, key) in zip(
        columns,
        categories,
    ):
        value = get_score(
            breakdown.get(key, 0)
        )

        with column:
            st.metric(
                label,
                f"{value}/100",
            )

    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    st.subheader("✅ Resume Strengths")

    strengths = get_list(
        analysis.get("strengths")
    )

    if strengths:
        for strength in strengths:
            st.markdown(f"- {strength}")
    else:
        st.write("No strengths were returned.")

    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    st.subheader("🛠️ Recommended Improvements")

    improvements = get_list(
        analysis.get("improvements")
    )

    if improvements:

        for index, improvement in enumerate(
            improvements,
            start=1,
        ):

            if isinstance(improvement, dict):

                priority = priority_label(
                    improvement.get(
                        "priority",
                        "Medium",
                    )
                )

                issue = improvement.get(
                    "issue",
                    "Improvement",
                )

                why = improvement.get(
                    "why_it_matters",
                    "",
                )

                recommendation = improvement.get(
                    "recommendation",
                    "",
                )

                example = improvement.get(
                    "example",
                    "",
                )

                with st.expander(
                    f"{index}. {priority} — {issue}",
                    expanded=(index <= 2),
                ):

                    if why:
                        st.markdown(
                            f"**Why it matters:** {why}"
                        )

                    if recommendation:
                        st.markdown(
                            f"**Recommendation:** {recommendation}"
                        )

                    if example:
                        st.markdown(
                            f"**Example:** {example}"
                        )

            else:
                st.markdown(
                    f"- {improvement}"
                )

    else:
        st.write(
            "No specific improvements were returned."
        )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    keyword_column, missing_column = st.columns(2)

    with keyword_column:

        st.subheader("🔑 Present Keywords")

        present_keywords = get_list(
            analysis.get(
                "present_keywords"
            )
        )

        if present_keywords:
            for keyword in present_keywords:
                st.markdown(
                    f"- `{keyword}`"
                )
        else:
            st.write(
                "No keywords were identified."
            )

    with missing_column:

        st.subheader("⚠️ Missing Keywords")

        missing_keywords = get_list(
            analysis.get(
                "missing_keywords"
            )
        )

        if missing_keywords:
            for keyword in missing_keywords:
                st.markdown(
                    f"- `{keyword}`"
                )
        else:
            st.write(
                "No major missing keywords were identified."
            )

    # --------------------------------------------------------
    # FORMATTING CHECKS
    # --------------------------------------------------------

    st.subheader("📋 Formatting Checks")

    formatting_checks = get_list(
        analysis.get(
            "formatting_checks"
        )
    )

    if formatting_checks:
        for item in formatting_checks:
            st.markdown(
                f"- {item}"
            )
    else:
        st.write(
            "No formatting checks were returned."
        )

    # --------------------------------------------------------
    # SECTION CHECKS
    # --------------------------------------------------------

    st.subheader("📑 Resume Section Checks")

    section_checks = get_list(
        analysis.get(
            "section_checks"
        )
    )

    if section_checks:
        for item in section_checks:
            st.markdown(
                f"- {item}"
            )
    else:
        st.write(
            "No section checks were returned."
        )

    # --------------------------------------------------------
    # DOWNLOAD JSON
    # --------------------------------------------------------

    st.divider()

    json_data = json.dumps(
        analysis,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "⬇️ Download Analysis as JSON",
        data=json_data,
        file_name="resume_ats_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ATS Resume Analyzer • AI-generated assessment. "
    "Always review suggestions and make sure your final resume "
    "accurately represents your real qualifications."
)
