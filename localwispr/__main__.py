"""Entry point for LocalWispr."""

from localwispr import __version__
from localwispr.config import load_config


def main() -> None:
    """Main entry point for LocalWispr."""
    print(f"LocalWispr v{__version__}")
    print("-" * 40)

    config = load_config()
    print(f"Model: {config['model']['name']}")
    print(f"Device: {config['model']['device']}")
    print()
    print("Configuration loaded successfully.")


if __name__ == "__main__":
    main()
