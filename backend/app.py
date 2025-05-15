from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import logging
import base64

import configs
from ml_models import load_models_and_tokenizer
from search import search_images

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models and tokenizer
device, tokenizer, resnet_model, vit_model = load_models_and_tokenizer()

# Request schema
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Search API. Use POST /search to search for images."}

@app.post("/search")
async def search(search_request: SearchRequest):
    query = search_request.query
    top_k = search_request.top_k

    resnet_response = []
    vit_response = []
    message = ""

    def image_to_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        resnet_results, _ = search_images(
            model=resnet_model,
            embeddings_path=configs.RESNET_EMBEDDINGS_PATH,
            query=query,
            tokenizer=tokenizer,
            device=device,
            model_path=configs.RESNET_MODEL_PATH,
            top_k=top_k
        )
        if resnet_results:
            resnet_response = [
                {"image": image_to_base64(image), "path": path, "similarity": float(similarity)}
                for image, path, similarity in resnet_results
            ]
        else:
            message += "No ResNet results. "
    except Exception as e:
        logging.warning(f"ResNet search failed: {str(e)}")
        message += f"ResNet error: {str(e)}. "

    try:
        vit_results, _ = search_images(
            model=vit_model,
            embeddings_path=configs.VIT_EMBEDDINGS_PATH,
            query=query,
            tokenizer=tokenizer,
            device=device,
            model_path=configs.VIT_MODEL_PATH,
            top_k=top_k
        )
        if vit_results:
            vit_response = [
                {"image": image_to_base64(image), "path": path, "similarity": float(similarity)}
                for image, path, similarity in vit_results
            ]
        else:
            message += "No ViT results. "
    except Exception as e:
        logging.warning(f"ViT search failed: {str(e)}")
        message += f"ViT error: {str(e)}. "

    if not resnet_response and not vit_response:
        return {"resnet": [], "vit": [], "message": message or "No matching images found."}

    return {"resnet": resnet_response, "vit": vit_response, "message": message}
