from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

HF_TOKEN = "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"

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)

# Define request body schema
class TextPair(BaseModel):
    model_answer: str
    student_answer: str

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to calculate similarity
def get_similarity_score(model_answer, student_answer):
    model_embedding = get_embeddings(model_answer)
    student_embedding = get_embeddings(student_answer)
    similarity_score = cosine_similarity(model_embedding, student_embedding)
    return similarity_score[0][0] * 100

# API route for similarity calculation
@app.post("/calculate-similarity/")
async def calculate_similarity(text_pair: TextPair):
    try:
        model_answer = text_pair.model_answer
        student_answer = text_pair.student_answer
        score = float(get_similarity_score(model_answer, student_answer))  # Convert to Python float
        return {"similarity_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
