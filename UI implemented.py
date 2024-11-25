import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Define labels
labels = ['Negative', 'Neutral', 'Positive']

# Function for sentiment analysis
def analyze_sentiment():
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Required", "Please enter text to analyze.")
        return

    # Preprocess input
    words = []
    for word in user_input.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        words.append(word)
    processed_input = " ".join(words)

    # Tokenize and predict
    encoded_input = tokenizer(processed_input, return_tensors='pt')
    output = model(**encoded_input)

    # Compute scores
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Display results
    result_text = "\n".join([f"{labels[i]}: {scores[i]:.2f}" for i in range(len(scores))])
    result_label.config(text=result_text)

# Create the Tkinter UI
root = tk.Tk()
root.title("Sentiment Analysis App")

# Input text widget
tk.Label(root, text="Enter text for sentiment analysis:").pack(pady=5)
text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(pady=5)

# Analyze button
analyze_button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Helvetica", 12), justify=tk.LEFT)
result_label.pack(pady=10)

# Run the Tkinter loop
root.mainloop()