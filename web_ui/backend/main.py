from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import io
import base64
from PIL import Image
import torch
from inference_wrapper import FractureModelManager
from chat_assistant import ChatAssistant, ChatRequest

# Initialize FastAPI app
app = FastAPI(title="FractureMamba-ViT API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Manager
model_manager = FractureModelManager()
chat_assistant = ChatAssistant()

@app.on_event("startup")
async def startup_event():
    print("Loading model into memory...")
    try:
        model_manager.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail=model_manager.load_error or "Classifier model is still unavailable. Check backend startup logs."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference in a thread so the event loop stays responsive
        # (prevents /health from timing out during long predictions)
        result = await asyncio.to_thread(model_manager.predict_and_explain, image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response_text = chat_assistant.get_response(request)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    model_loaded = model_manager.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "load_error": model_manager.load_error,
        "used_swin_fallback": model_manager.used_swin_fallback,
        "yolo_loaded": model_manager.yolo_model is not None,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
