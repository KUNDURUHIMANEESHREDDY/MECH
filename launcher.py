import multiprocessing
import uvicorn

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("=" * 60)
    print("  MECH: Mechanistic Interpretability & Out-of-Core Engine")
    print("  Server running on: http://127.0.0.1:8000")
    print("  Swagger Docs:      http://127.0.0.1:8000/docs")
    print("=" * 60)
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
