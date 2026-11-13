"""Start the local MedLens API; models are loaded only for accepted jobs."""

from medlens.api_server import main


if __name__ == "__main__":
    raise SystemExit(main())
