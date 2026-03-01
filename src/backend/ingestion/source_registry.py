"""Registry mapping source names to their scraper classes and configuration keys."""

from ingestion.scrapers import *


class SourceRegistry:
    """Central registry for supported ingestion sources.

    Maps source name strings to their corresponding scraper classes and
    environment variable keys used to configure them.

    Attributes:
        SOURCES (dict[str, dict]): Maps source names to a config dict with keys:
            - ``"env_var"`` (str): The environment variable used to configure the source.
            - ``"scraper_class"`` (type): The scraper class for the source.
    """

    SOURCES_MAPPING = {
        "medium": {
            "env_var": "MEDIUM_USERNAME",
            "scraper_class": MediumScraper,
        },
        "github": {
            "env_var": "GITHUB_USERNAME",
            "scraper_class": GitHubScraper,
        },
        "resume": {
            "env_var": "RESUME_URL",
            "scraper_class": ResumeScraper,
        },
    }

    @staticmethod
    def get_sources():
        """Return the list of supported source names."""
        return list(SourceRegistry.SOURCES_MAPPING.keys())
    
    @staticmethod
    def get_env_var(source_name):
        """Return the environment variable key associated with the given source name.

        Args:
            source_name (str): The name of the source (e.g. ``"medium"``).

        Returns:
            str: The environment variable key corresponding to ``source_name``.

        Raises:
            KeyError: If ``source_name`` does not match a registered source.
        """
        return SourceRegistry.SOURCES_MAPPING[source_name]["env_var"]

    @staticmethod
    def get_scraper_class(source_name):
        """Return the scraper class associated with the given source name.

        Performs the lookup at call time so that mocks applied via
        ``unittest.mock.patch`` are respected in tests.

        Args:
            source_name (str): The name of the source (e.g. ``"medium"``).

        Returns:
            type: The scraper class corresponding to ``source_name``.

        Raises:
            KeyError: If ``source_name`` does not match a registered source.
        """

        return SourceRegistry.SOURCES_MAPPING[source_name]["scraper_class"]

