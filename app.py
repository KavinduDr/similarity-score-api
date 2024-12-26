from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer from local directory
local_model_path = "./scoreModel/model"  # Adjust the path to match your folder structure
local_tokenizer_path = "./scoreModel/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
model = AutoModel.from_pretrained(local_model_path)

# Define the data structure for the request
class SimilarityRequest(BaseModel):
    model_answer: str
    student_answer: str

# Function to calculate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to calculate similarity score
def get_similarity_score(model_answer, student_answer):
    model_embedding = get_embeddings(model_answer)
    student_embedding = get_embeddings(student_answer)
    similarity_score = cosine_similarity(model_embedding, student_embedding)
    return similarity_score[0][0] * 100

@app.get("/")
async def root():
    return {"message": "Welcome to the similarity calculator!"}

# Endpoint to calculate similarity
@app.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    try:
        score = float(get_similarity_score(request.model_answer, request.student_answer))
        return {"similarity_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
