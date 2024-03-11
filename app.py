from transformers import pipeline
import pandas as pd
import streamlit as st

tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Load the dataset
dataset = pd.read_csv("Conversation.csv")

# Assuming the dataset has 'question' and 'answer' columns
table = pd.DataFrame({'Question': dataset['question'], 'Answer': dataset['answer']})

table = table.astype(str)
table = table.fillna('')

# Assuming you have a DataFrame named 'df' with your dataset
data = {
    'index': [0, 1, 2, 3, 4],
    'Unnamed: 0': [0, 1, 2, 3, 4],
    'question': ["hi", "how are you doing?", "i am fine. how about yourself?", "no problem. so how have you been?",
                 "i've been great. what about you?"],
    'answer': ["hello", "i'm fine. how about yourself?", "no problem. so how have you been?",
               "i've been great. what about you?", "i've been good. i'm in school right now."]
}

df = pd.DataFrame(data)

while True:
    # Get user input for the question
    question = st.text_input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == 'exit':
        break

    # Generate the answer using TAPAS
    answer = tqa(table=table, query=question)["answer"]

    # Print the result
    st.write(f"Question: {question}\nAnswer: {answer}\n")
