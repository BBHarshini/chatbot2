from transformers import pipeline
import pandas as pd
import streamlit as st

# Set up TAPAS model pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Sample data
data = {
    'index': [0, 1, 2, 3, 4],
    'Unnamed: 0': [0, 1, 2, 3, 4],
    'question': ["hi", "how are you doing?", "i am fine. how about yourself?", "no problem. so how have you been?",
                 "i've been great. what about you?"],
    'answer': ["hello", "i'm fine. how about yourself?", "no problem. so how have you been?",
               "i've been great. what about you?", "i've been good. i'm in school right now."]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function for TAPAS model logic
def tqa(table, query):
    # Replace this with your TAPAS model logic
    return {"answer": table['answer'].values[0]}

# Streamlit app
def main():
    st.title("TAPAS Model QA App")
    st.write("Ask a question and get an answer!")

    # Get user input for the question
    question = st.text_input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == 'exit':
        st.stop()

    # Find the corresponding row in the DataFrame
    row = df[df['question'] == question]

    # If the question is found in the dataset, generate the answer
    if not row.empty:
        answer = tqa(table=row, query=question)["answer"]
        st.write(f"Question: {question}\nAnswer: {answer}\n")
    else:
        st.write(f"Hello, have a good day.\n")

if __name__ == "__main__":
    main()
