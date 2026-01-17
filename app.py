import streamlit as st

from rag_retrieval import (
    is_valid_scheme_query,
    retrieve_context,
    refine_context
)
from answer_generator import generate_answer
from llm_client import call_llm


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SustainaGov AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 SustainaGov AI")
st.subheader("AI Assistant for Government Sustainability Schemes")

st.write(
    "Ask questions about Indian government schemes related to "
    "clean energy, water conservation, and sustainability."
)

# -------------------- USER INPUT --------------------
user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g. solar subsidy for home"
)

language = st.selectbox(
    "Select language:",
    ["English"]  # Future-ready
)

ask_button = st.button("Ask")

# -------------------- RESPONSE AREA --------------------
if ask_button:
    if not user_query.strip():
        st.warning("Please enter a question.")
    
    elif not is_valid_scheme_query(user_query):
        st.error(
            "Invalid question.\n\n"
            "Please ask about a government scheme.\n"
            "Example: *solar subsidy for home*"
        )
    
    else:
        with st.spinner("Finding the best answer for you..."):
            results = retrieve_context(user_query)
            clean_context = refine_context(results)

            answer_prompt = generate_answer(user_query, clean_context)
            final_answer = call_llm(answer_prompt)

        st.markdown("---")
        st.markdown("### 🤖 Answer")
        st.markdown(final_answer, unsafe_allow_html=False)
        st.caption(
                "📌 **Source:** Official Government of India portals "
                "(e.g., Ministry of New and Renewable Energy, Ministry of Jal Shakti)"
            )

