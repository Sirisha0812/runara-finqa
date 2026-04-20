"""vLLM server for FinQA chatbot with GPU/MPS support.

NOTE: vLLM requires CUDA (NVIDIA GPUs) and does NOT work on Mac.
For Mac users, use src/ollama_server.py instead.
"""

import argparse
import subprocess
import sys
from typing import Optional

from src.config import config
from src.logger import LoggerContext, get_logger

logger = get_logger(__name__)


def check_gpu_availability() -> tuple[str, Optional[str]]:
    """
    Check GPU availability and return appropriate device type.

    Returns:
        Tuple of (device_type, error_message)
        device_type: "cuda", "mps", or "cpu"
        error_message: None if successful, error string otherwise
    """
    import platform

    # Check if CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(
                "cuda_detected",
                gpu_count=gpu_count,
                gpu_name=gpu_name,
            )
            return "cuda", None
    except Exception as e:
        logger.warning("cuda_check_failed", error=str(e))

    # Check if MPS (Apple Metal) is available
    if platform.system() == "Darwin":  # macOS
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("mps_detected", device="Apple Silicon GPU")
                return "mps", None
        except Exception as e:
            logger.warning("mps_check_failed", error=str(e))

    # Fallback to CPU
    logger.warning(
        "no_gpu_detected",
        message="No GPU detected. vLLM will run on CPU (very slow).",
    )
    return "cpu", "No GPU detected - performance will be severely degraded"


def start_vllm_server(
    model: Optional[str] = None,
    port: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.90,
    device: Optional[str] = None,
) -> None:
    """
    Start vLLM server with configuration from config.py.

    Args:
        model: Model name (default from config)
        port: Port number (default from config)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default from config)
        max_model_len: Maximum model context length (default: 4096)
        gpu_memory_utilization: GPU memory utilization (default: 0.90)
        device: Force device type ("cuda", "mps", "cpu")
    """
    with LoggerContext(logger, "start_vllm_server"):
        # Get configuration
        model_name = model or config.vllm.model
        server_port = port or config.vllm.port
        tp_size = tensor_parallel_size or config.vllm.tensor_parallel_size
        max_len = max_model_len or 4096

        # Detect device if not specified
        if device is None:
            device, error = check_gpu_availability()
            if error:
                logger.warning("device_fallback", device=device, warning=error)
        else:
            logger.info("device_forced", device=device)

        # Build vLLM command
        cmd = [
            "vllm",
            "serve",
            model_name,
            "--port",
            str(server_port),
            "--max-model-len",
            str(max_len),
            "--trust-remote-code",
        ]

        # Add device-specific arguments
        if device == "cuda":
            cmd.extend(
                [
                    "--tensor-parallel-size",
                    str(tp_size),
                    "--gpu-memory-utilization",
                    str(gpu_memory_utilization),
                ]
            )
            logger.info(
                "cuda_config",
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        elif device == "mps":
            # vLLM doesn't natively support MPS, but we can try CPU mode
            logger.warning(
                "mps_limitation",
                message="vLLM does not natively support Apple MPS. Falling back to CPU mode.",
                recommendation="Consider using a CUDA-enabled GPU for better performance.",
            )
            cmd.extend(["--device", "cpu"])
            device = "cpu"
        elif device == "cpu":
            cmd.extend(["--device", "cpu"])
            logger.warning(
                "cpu_mode_warning",
                message="Running on CPU - inference will be very slow (30-60s per response).",
                recommendation="Use a CUDA-enabled GPU for production workloads.",
            )

        # Log configuration
        logger.info(
            "vllm_server_starting",
            model=model_name,
            port=server_port,
            device=device,
            max_model_len=max_len,
            api_base=config.vllm.api_base,
        )

        # Print user-friendly message
        print("\n" + "=" * 80)
        print("vLLM Server Startup")
        print("=" * 80)
        print(f"\nModel: {model_name}")
        print(f"Device: {device.upper()}")
        print(f"Port: {server_port}")
        print(f"API Base: http://localhost:{server_port}/v1")
        print(f"Max Model Length: {max_len} tokens")
        if device == "cuda":
            print(f"Tensor Parallel Size: {tp_size}")
            print(f"GPU Memory Utilization: {gpu_memory_utilization * 100:.0f}%")
        elif device == "cpu":
            print(
                "\n⚠️  WARNING: Running on CPU - inference will be very slow (30-60s per response)."
            )
            print(
                "   For production use, please run on a CUDA-enabled GPU with at least 48GB VRAM."
            )

        print("\nStarting server...\n")
        print("=" * 80)

        # Start vLLM server
        try:
            # Run vLLM server and stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output and watch for "ready" message
            server_ready = False
            for line in iter(process.stdout.readline, ""):
                print(line, end="")

                # Detect when server is ready
                if not server_ready and (
                    "Uvicorn running on" in line or "Application startup complete" in line
                ):
                    server_ready = True
                    print("\n" + "=" * 80)
                    print("✅ vLLM SERVER READY")
                    print("=" * 80)
                    print(f"\nServer running at: http://localhost:{server_port}/v1")
                    print(f"Model: {model_name}")
                    print(f"Device: {device.upper()}")
                    print("\nYou can now run the agent with:")
                    print("  python -m src.agent")
                    print("\nTo stop the server, press Ctrl+C")
                    print("=" * 80 + "\n")
                    logger.info(
                        "vllm_server_ready",
                        port=server_port,
                        model=model_name,
                        device=device,
                    )

            # Wait for process to complete
            process.wait()

            if process.returncode != 0:
                logger.error(
                    "vllm_server_failed",
                    return_code=process.returncode,
                )
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("Shutting down vLLM server...")
            print("=" * 80)
            logger.info("vllm_server_stopped", reason="keyboard_interrupt")
            if process:
                process.terminate()
                process.wait()
            sys.exit(0)

        except FileNotFoundError:
            logger.error(
                "vllm_not_found",
                message="vLLM executable not found. Make sure vLLM is installed.",
                installation_command="pip install vllm>=0.6.0",
            )
            print("\n❌ ERROR: vLLM not found")
            print("Please install vLLM with: pip install vllm>=0.6.0")
            sys.exit(1)

        except Exception as e:
            logger.error("vllm_server_error", error=str(e), error_type=type(e).__name__)
            print(f"\n❌ ERROR: {e}")
            sys.exit(1)


def main():
    """Main entry point for vLLM server."""
    parser = argparse.ArgumentParser(
        description="Start vLLM server for FinQA chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default config from .env
  python -m src.vllm_server

  # Start with custom model
  python -m src.vllm_server --model Qwen/Qwen2.5-7B-Instruct

  # Start on custom port
  python -m src.vllm_server --port 8001

  # Force CPU mode (for testing)
  python -m src.vllm_server --device cpu

  # Custom GPU memory utilization
  python -m src.vllm_server --gpu-memory-utilization 0.8
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name (default: {config.vllm.model})",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port number (default: {config.vllm.port})",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=f"Number of GPUs for tensor parallelism (default: {config.vllm.tensor_parallel_size})",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization fraction (default: 0.90)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force device type (default: auto-detect)",
    )

    args = parser.parse_args()

    # Start server
    start_vllm_server(
        model=args.model,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=args.device,
    )


if __name__ == "__main__":
    main()
