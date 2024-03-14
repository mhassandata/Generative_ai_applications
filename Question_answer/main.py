import streamlit as st
from transformers import pipeline

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="mhassan101/bert-finetuned-squad-ds", tokenizer="mhassan101/bert-finetuned-squad-ds")

def get_answer(context, question):
    return qa_pipeline({'context': context, 'question': question})

def main():
    st.title("Question Answering System")

    # Context input area
    context = st.text_area("Context", "Enter the context here...")

    # Question input area
    question = st.text_input("Question", "Enter your question here...")

    # Button to get answer
    if st.button("Get Answer"):
        if context and question:
            # Get answer from the model
            answer = get_answer(context, question)
            st.write("Answer:", answer['answer'])
        else:
            st.write("Please enter both a context and a question.")

if __name__ == "__main__":
    main()
