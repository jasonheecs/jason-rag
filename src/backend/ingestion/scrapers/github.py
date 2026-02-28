"""GitHub scraper for fetching profile and repository information."""
import requests
from typing import List, Dict, Optional
from datetime import datetime


class GitHubScraper:
    """Scrapes GitHub profile and repositories via GitHub API."""

    API_BASE_URL = "https://api.github.com"
    SOURCE_NAME = "github"

    def __init__(self, username: str, token: Optional[str] = None):
        """
        Initialize GitHub scraper.

        Args:
            username: GitHub username to scrape
            token: Optional GitHub personal access token for higher rate limits
        """
        self.username = username
        self.token = token
        self.session = self._create_session()

    def scrape(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch user profile and repositories, filtering by last_scraped_date."""
        documents = []

        # Scrape profile info
        profile_doc = self._scrape_profile()
        if profile_doc:
            if last_scraped_date is None or profile_doc['published_date'] > last_scraped_date:
                documents.append(profile_doc)

        # Scrape repositories
        repo_docs = self._scrape_repositories(last_scraped_date)
        documents.extend(repo_docs)

        print(f"Scraped {len(documents)} GitHub documents")
        return documents

    def _create_session(self) -> requests.Session:
        """Create HTTP session with authentication if token provided."""
        session = requests.Session()
        if self.token:
            session.headers.update({"Authorization": f"token {self.token}"})
        return session

    def _scrape_profile(self) -> Optional[Dict]:
        """Fetch and parse user profile information."""
        url = f"{self.API_BASE_URL}/users/{self.username}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            user_data = response.json()
            return self._parse_profile(user_data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching GitHub profile: {e}")
            return None

    def _scrape_repositories(self, last_scraped_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch and parse user repositories."""
        repos = []
        page = 1
        per_page = 100

        while True:
            url = f"{self.API_BASE_URL}/users/{self.username}/repos"
            params = {
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                repo_list = response.json()

                if not repo_list:
                    break

                for repo_data in repo_list:
                    repo_doc = self._parse_repository(repo_data)

                    # Stop if we've reached repos older than last_scraped_date
                    if last_scraped_date and repo_doc['published_date'] < last_scraped_date:
                        return repos

                    repos.append(repo_doc)

                page += 1

            except requests.exceptions.RequestException as e:
                print(f"Error fetching GitHub repositories: {e}")
                break

        return repos

    def _parse_profile(self, user_data: Dict) -> Dict:
        """Transform user API response into document dictionary."""
        profile_parts = self._extract_profile_parts(user_data)
        content = self._build_content(profile_parts)
        metadata = self._build_profile_metadata(user_data)

        return self._create_document(
            title=f"GitHub Profile: {user_data['name'] or user_data['login']}",
            content=content,
            url=user_data['html_url'],
            published_date=self._parse_iso_date(user_data['updated_at']),
            metadata=metadata
        )

    def _parse_repository(self, repo_data: Dict) -> Dict:
        """Transform repository API response into document dictionary."""
        repo_parts = self._extract_repo_parts(repo_data)
        content = self._build_content(repo_parts)
        metadata = self._build_repo_metadata(repo_data)

        return self._create_document(
            title=f"Repository: {repo_data['name']}",
            content=content,
            url=repo_data['html_url'],
            published_date=self._parse_iso_date(repo_data['updated_at']),
            metadata=metadata
        )

    def _build_content(self, parts: List[str]) -> str:
        """Join content parts into a single string."""
        return " | ".join(parts)

    def _parse_iso_date(self, iso_string: str) -> datetime:
        """Parse ISO 8601 date string to datetime."""
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))

    def _create_document(
        self,
        title: str,
        content: str,
        url: str,
        published_date: datetime,
        metadata: Dict,
    ) -> Dict:
        """Create a document dictionary with standard structure."""
        return {
            'title': title,
            'content': content,
            'url': url,
            'published_date': published_date,
            'source': self.SOURCE_NAME,
            'metadata': metadata,
        }

    def _extract_profile_parts(self, user_data: Dict) -> List[str]:
        """Extract profile content parts."""
        parts = [
            f"Username: {user_data['login']}",
            f"Name: {user_data.get('name') or 'N/A'}",
        ]

        if user_data.get('bio'):
            parts.append(f"Bio: {user_data['bio']}")
        if user_data.get('company'):
            parts.append(f"Company: {user_data['company']}")
        if user_data.get('location'):
            parts.append(f"Location: {user_data['location']}")
        if user_data.get('blog'):
            parts.append(f"Blog: {user_data['blog']}")

        parts.extend([
            f"Public Repositories: {user_data['public_repos']}",
            f"Followers: {user_data['followers']}",
            f"Following: {user_data['following']}",
            f"Created at: {user_data['created_at']}",
        ])

        return parts

    def _extract_repo_parts(self, repo_data: Dict) -> List[str]:
        """Extract repository content parts."""
        parts = [f"Repository: {repo_data['name']}"]

        if repo_data.get('description'):
            parts.append(f"Description: {repo_data['description']}")

        parts.extend([
            f"Language: {repo_data.get('language') or 'N/A'}",
            f"Stars: {repo_data['stargazers_count']}",
            f"Forks: {repo_data['forks_count']}",
            f"Open Issues: {repo_data['open_issues_count']}",
            f"Topics: {', '.join(repo_data.get('topics', []) or [])}",
            f"URL: {repo_data['html_url']}",
        ])

        return parts

    def _build_profile_metadata(self, user_data: Dict) -> Dict:
        """Build metadata dictionary for user profile."""
        return {
            'type': 'profile',
            'login': user_data['login'],
            'followers': user_data['followers'],
            'public_repos': user_data['public_repos'],
            'company': user_data.get('company'),
            'location': user_data.get('location'),
        }

    def _build_repo_metadata(self, repo_data: Dict) -> Dict:
        """Build metadata dictionary for repository."""
        return {
            'type': 'repository',
            'repo_name': repo_data['name'],
            'language': repo_data.get('language'),
            'stars': repo_data['stargazers_count'],
            'forks': repo_data['forks_count'],
            'is_fork': repo_data['fork'],
        }
