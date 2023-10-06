import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Load the pre-trained LLM model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Create a Streamlit app
st.title("Legal Solution Generator")

# Add a text input field for the user to enter their legal problem
user_input = st.text_input("Enter your legal problem:")

# Generate legal-based solutions for the user's problem
if user_input:
    # Add the user's input to the model prompt
    prompt = "Legal problem: " + user_input + "\nLegal solution:"
    # Generate the solution using the model
    output = model.generate(
        input_ids=tokenizer.encode(prompt, return_tensors="pt"),
        max_length=1000,
        temperature=0.7,
    )
    # Decode the generated solution
    solution = tokenizer.decode(output[0], skip_special_tokens=True)
    # Display the generated solution in a table format
    df = pd.DataFrame({"Legal Solution": [solution]})
    st.table(df)
    
