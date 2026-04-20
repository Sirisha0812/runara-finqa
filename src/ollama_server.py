"""Ollama server for FinQA chatbot - Mac-compatible alternative to vLLM."""

import argparse
import subprocess
import sys
import time
from typing import Optional

import requests

from src.config import config
from src.logger import LoggerContext, get_logger

logger = get_logger(__name__)


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_ollama_running() -> bool:
    """Check if Ollama server is already running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """List available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception:
        return []


def pull_model(model_name: str) -> bool:
    """Pull an Ollama model if not already available."""
    with LoggerContext(logger, "pull_ollama_model", model=model_name):
        try:
            print(f"\n📥 Pulling model: {model_name}")
            print("This may take a few minutes for large models...\n")

            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(process.stdout.readline, ""):
                print(line, end="")

            process.wait()

            if process.returncode == 0:
                logger.info("model_pulled_successfully", model=model_name)
                print(f"\n✅ Model {model_name} pulled successfully!")
                return True
            else:
                logger.error("model_pull_failed", model=model_name, return_code=process.returncode)
                return False

        except Exception as e:
            logger.error("model_pull_error", model=model_name, error=str(e))
            print(f"\n❌ Error pulling model: {e}")
            return False


def start_ollama_server(
    model: Optional[str] = None,
    auto_pull: bool = True,
) -> None:
    """
    Start or verify Ollama server and ensure model is available.

    Args:
        model: Model name (e.g., "qwen2.5:32b", "deepseek-r1:32b")
        auto_pull: Automatically pull model if not available
    """
    with LoggerContext(logger, "start_ollama_server"):
        # Check if Ollama is installed
        if not check_ollama_installed():
            logger.error("ollama_not_installed")
            print("\n❌ ERROR: Ollama is not installed")
            print("\nTo install Ollama:")
            print("  1. Visit: https://ollama.com/download")
            print("  2. Download and install Ollama for macOS")
            print("  3. Run: ollama serve")
            print("\nAlternatively, use Homebrew:")
            print("  brew install ollama")
            sys.exit(1)

        # Check if Ollama server is running
        if not check_ollama_running():
            logger.info("starting_ollama_server")
            print("\n🚀 Starting Ollama server...")
            print("If this fails, start it manually with: ollama serve\n")

            try:
                # Start Ollama server in background
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait for server to start
                for i in range(10):
                    time.sleep(1)
                    if check_ollama_running():
                        logger.info("ollama_server_started")
                        print("✅ Ollama server started successfully!")
                        break
                else:
                    logger.error("ollama_server_start_timeout")
                    print("\n❌ ERROR: Ollama server failed to start")
                    print("Please start it manually with: ollama serve")
                    sys.exit(1)

            except Exception as e:
                logger.error("ollama_server_start_error", error=str(e))
                print(f"\n❌ ERROR starting Ollama server: {e}")
                print("Please start it manually with: ollama serve")
                sys.exit(1)
        else:
            logger.info("ollama_server_already_running")
            print("\n✅ Ollama server is already running")

        # Determine model to use
        model_name = model or "qwen2.5:32b"  # Default to Qwen2.5 32B

        # Check if model is available
        available_models = list_ollama_models()
        logger.info("available_models", models=available_models)

        if model_name not in available_models:
            print(f"\n⚠️  Model '{model_name}' is not available locally")
            print(f"Available models: {', '.join(available_models) if available_models else 'None'}")

            if auto_pull:
                if not pull_model(model_name):
                    print("\n❌ Failed to pull model")
                    print(f"You can pull it manually with: ollama pull {model_name}")
                    sys.exit(1)
            else:
                print(f"\nTo pull the model manually:")
                print(f"  ollama pull {model_name}")
                sys.exit(1)
        else:
            logger.info("model_available", model=model_name)
            print(f"\n✅ Model '{model_name}' is available")

        # Print summary
        print("\n" + "=" * 80)
        print("✅ OLLAMA SERVER READY")
        print("=" * 80)
        print(f"\nServer running at: http://localhost:11434")
        print(f"Model: {model_name}")
        print(f"Device: Apple Silicon (MPS)")
        print("\nOllama API is OpenAI-compatible at:")
        print("  http://localhost:11434/v1")
        print("\nTo use with the agent:")
        print("  1. Update .env: VLLM_API_BASE=http://localhost:11434/v1")
        print(f"  2. Update .env: VLLM_MODEL={model_name}")
        print("  3. Run: python -m src.agent")
        print("\nTo stop Ollama:")
        print("  pkill ollama")
        print("=" * 80 + "\n")

        logger.info(
            "ollama_server_ready",
            model=model_name,
            api_base="http://localhost:11434/v1",
        )


def main():
    """Main entry point for Ollama server."""
    parser = argparse.ArgumentParser(
        description="Start Ollama server for FinQA chatbot (Mac-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default model (qwen2.5:32b)
  python -m src.ollama_server

  # Start with specific model
  python -m src.ollama_server --model qwen2.5:7b

  # Use DeepSeek-R1
  python -m src.ollama_server --model deepseek-r1:32b

  # Don't auto-pull missing models
  python -m src.ollama_server --no-auto-pull

Recommended models for FinQA (numerical reasoning):
  - qwen2.5:32b (best quality, needs ~20GB RAM)
  - qwen2.5:7b (faster, needs ~8GB RAM)
  - deepseek-r1:32b (strong reasoning, needs ~20GB RAM)
  - qwen2.5:14b (balanced, needs ~12GB RAM)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: qwen2.5:32b)",
    )

    parser.add_argument(
        "--no-auto-pull",
        action="store_true",
        help="Don't automatically pull missing models",
    )

    args = parser.parse_args()

    # Start server
    start_ollama_server(
        model=args.model,
        auto_pull=not args.no_auto_pull,
    )


if __name__ == "__main__":
    main()
