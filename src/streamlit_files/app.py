import streamlit as st
from PIL import Image
from src.dependency_functions.functions import *
from src.dependency_functions.run_code import *
from src.utils.logger import log_info, log_error
import asyncio

st.set_page_config(page_title="StatSolveR")
st.title("StatSolveR")
st.write("An LLM-Powered Python Executor for Statistical Problem Solving")

if 'query' not in st.session_state:
    st.session_state.query = False

with st.sidebar:
    st.image(Image.open('src/streamlit_files/StatSolveR_logo.png'))
    suggestions = [
    "For x(independent variable) = [1, 2, 3] and y(dependent variable) = [2, 4, 6] Give me the formula to calculate linear regression using python and predict value for x = 5.",
    "Given the dataset: [10, 20, 30, 40, 50], calculate the mean, median, and standard deviation using Python.",
    "For x = [1, 2, 3, 4, 5] and y = [2, 4, 6, 8, 10], calculate the Pearson correlation coefficient using Python."
    ]
    st.write("")
    st.write("")
    st.write("Type a question below:")
    # Dropdown for suggestions
    selected = st.selectbox("Or pick a suggestion:", [""] + suggestions)
    # Text input that fills based on selection
    query = st.text_input("Enter your question:", selected)
    if query:
        st.session_state.query = True

def sync_wrapper():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(ask_llm(query))

if st.session_state.query:
    if query:
        log_info(f"User Query: {query}")
        try:
            result = ask_llm(query)
            code, output = run_code(result)
            st.write("**Given Query:**")
            st.write(query)
            st.write("**Execution Results:**")
            st.markdown(output, unsafe_allow_html=True)
            with st.expander("Code Generated by LLM"):
                st.write("**Response:**")
                st.code(code, language="python")
            
            log_info(f"User Query: {query} | Response: {result[:50]}... | Agent Out: {output}")  # Log preview
        except Exception as e:
            log_error(f"Error processing query: {e}")
            st.write("An error occurred. Please try again.")
    else:
        st.warning("Please enter a query!")