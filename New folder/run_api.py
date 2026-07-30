"""
Entry point for running the Neuron Inspector API.

Usage:
    python run_api.py [--host HOST] [--port PORT]

The API will be available at http://localhost:8000
Documentation at http://localhost:8000/docs
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Run the Neuron Inspector API"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes",
    )
    args = parser.parse_args()

    print(f"Starting Neuron Inspector API on {args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/docs")
    print(f"Redoc: http://{args.host}:{args.port}/redoc")

    uvicorn.run(
        "backend.interpretability.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
