"""
Entry point.
Loads .env, then starts Uvicorn.

Usage:
    python run.py
"""
from dotenv import load_dotenv
load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
