"""
NeuroTriage FastAPI Backend

Main entry point for the NeuroTriage FastAPI application.
Initializes the app, loads environment variables, and sets up routes.

Environment Variables:
    GROQ_API_KEY: API key for Groq LLM provider
    GEMINI_API_KEY: API key for Google Gemini LLM provider
    OLLAMA_MODEL: Model name to use with local Ollama instance
    GROQ_FAST_MODEL: Fast model variant for Groq (optional)
    OLLAMA_EMBED_MODEL: Embedding model for Ollama (optional)

Usage:
    conda run -n neurotriage-env uvicorn app.main:app --reload
    
This will start the FastAPI development server on http://localhost:8000
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NeuroTriage API",
    description="Multi-agent brain MRI analysis system with LangGraph orchestration",
    version="0.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
from app.db.models import init_db
init_db()

# Load VGG16 model on startup
from app.models.vision_loader import load_vgg16_model
try:
    # In development, allow mock model as fallback for faster iteration
    use_mock = os.getenv("NEUROTRIAGE_DEV_MODE", "false").lower() == "true"
    model = load_vgg16_model(use_mock_on_failure=use_mock)
    print("✓ VGG16 model loaded successfully on startup")
except Exception as e:
    print(f"⚠ Warning: VGG16 model failed to load on startup: {e}")
    print("  The model will be loaded on first request with HTTP 503 fallback")

# Import and include routers
from app.routes.analyze import router as analyze_router
app.include_router(analyze_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "NeuroTriage API is running",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    env_vars = {
        "GROQ_API_KEY": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "GEMINI_API_KEY": "configured" if os.getenv("GEMINI_API_KEY") else "missing",
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "not set"),
    }
    return {
        "status": "healthy",
        "environment": env_vars,
        "database": "initialized"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
