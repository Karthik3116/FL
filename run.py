"""Entry point for the Federated Learning Inference Console.

Run with:
    python run.py
or:
    flask --app app run
"""
import argparse

from app import create_app


def main():
    parser = argparse.ArgumentParser(description="FL Inference Flask Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="Port to bind (default: 5005)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    application = create_app()
    application.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
