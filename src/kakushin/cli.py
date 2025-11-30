"""Command-line interface for Kakushin."""

import argparse
import sys

import uvicorn


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="kakushin",
        description="Long-form AI Video Generation Agent",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a video")
    gen_parser.add_argument(
        "scenario",
        help="Scenario/story description or path to YAML file",
    )
    gen_parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target duration in seconds (default: 60)",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )

    # GPU info command
    subparsers.add_parser("gpu", help="Show GPU information")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "server":
        uvicorn.run(
            "kakushin.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    elif args.command == "generate":
        print(f"Generating video from scenario: {args.scenario}")
        print(f"Duration: {args.duration}s")
        print(f"Output: {args.output}")
        # TODO: Implement video generation CLI
        print("Video generation CLI not yet implemented")
        return 1

    elif args.command == "gpu":
        from kakushin.utils.gpu_monitor import GPUMonitor

        monitor = GPUMonitor()
        stats = monitor.get_stats()

        print("GPU Information")
        print("=" * 40)
        print(f"Available: {stats.available}")
        if stats.available:
            print(f"Device: {stats.device_name}")
            print(f"Total Memory: {stats.total_memory_gb:.2f} GB")
            print(f"Allocated: {stats.allocated_memory_gb:.2f} GB")
            print(f"Reserved: {stats.reserved_memory_gb:.2f} GB")
            print(f"Free: {stats.free_memory_gb:.2f} GB")
            print(f"Usage: {stats.usage_percent:.1f}%")
            print(f"Under 92% threshold: {'Yes' if stats.is_under_threshold else 'No'}")
        return 0

    elif args.command == "version":
        from kakushin import __version__

        print(f"Kakushin v{__version__}")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
