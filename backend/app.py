"""
Wrapper for Hugging Face Spaces deployment
This file imports the FastAPI app from main.py
"""
from main import app

# This is needed for Hugging Face Spaces to find the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
