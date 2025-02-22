import streamlit as st
from PIL import Image
from src.dependency_function.functions import ask_llm
from src.utils.logger import log_info, log_error

st.title("📊 Statistics AI Assistant")
st.write("Ask a question related to statistics, and I'll fetch an answer!")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        log_info(f"User Query: {query}")
        try:
            response, images = ask_llm(query)
            st.write("### Answer:")
            st.write(response)
            
            if images:
                st.write("### Related Images:")
                for img in images:
                    img_path = f"src/data/raw/image/{img['file']}"
                    st.image(Image.open(img_path), caption=f"{img['caption']} (Label: {img['label']})", use_column_width=True)

            log_info("Response & images displayed successfully.")
        except Exception as e:
            log_error(f"Error in Streamlit query: {e}")
            st.error("An error occurred while fetching the answer.")
    else:
        st.warning("Please enter a query!")