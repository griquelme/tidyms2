"""Download example datasets."""

import pathlib
from typing import Any

import httpx
import pydantic

from ..core.exceptions import DatasetNotFound


class DatasetFileContent(pydantic.BaseModel):
    """Store a dataset file content."""

    name: str
    """The file name"""

    path: str
    """The file path in the GitHub repo"""

    content: str | None = None


def download_dataset(name: str, download_dir: pathlib.Path | str, token: str | None = None):
    """Download a list of files from the `data repository <https://github.com/griquelme/tidyms-data>`_.

    :param name: name of the data directory
    :param download_dir : path to download the data.
    :param token: A GitHub personal access token

    Examples
    --------
    Download the `data.csv` file from the `reference-materials` directory into
    the current directory:

    >>> import tidyms as ms
    >>> dataset = "reference-materials"
    >>> ms.fileio.download_tidyms_data(dataset, file_list, download_dir=".")

    See Also
    --------
    download_dataset
    load_dataset

    """
    owner = "griquelme"
    repo = "tidyms-data"

    dataset_files = list_dataset_files(owner, repo, name, token=token)

    if isinstance(download_dir, str):
        download_dir = pathlib.Path(download_dir)

    download_dir.mkdir(exist_ok=True, parents=True)

    for file_content in dataset_files:
        file_path = download_dir / file_content.name
        if file_path.is_file():
            continue

        if file_content.content is None:
            file_content.content = _get_github_repo_contents(owner, repo, file_content.path, raw=True, token=token)

        file_path.write_text(file_content.content, newline="")  # type: ignore


def get_dataset_file(dataset: str, file: str, download_path: pathlib.Path) -> pathlib.Path:
    """Get the path to an example dataset file."""
    download_dataset(dataset, download_path)

    path = download_path / dataset / file

    if not path.is_file():
        msg = f"File {file} not found in dataset {dataset}."
        raise DatasetNotFound(msg)

    return path


def list_dataset_files(owner: str, repo: str, dataset: str, token: str | None = None) -> list[DatasetFileContent]:
    """List the files in a directory in a GitHub repository."""
    try:
        contents = _get_github_repo_contents(owner, repo, dataset, raw=False, token=token)
    except (httpx.HTTPError, ValueError):
        raise DatasetNotFound(dataset)

    if not isinstance(contents, list):
        raise DatasetNotFound(dataset)

    return [DatasetFileContent(**x) for x in contents]


def _get_github_repo_contents(owner: str, repo: str, path: str, raw: bool = False, token: str | None = None) -> Any:
    endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github.raw+json"} if raw else dict()
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    response = httpx.get(endpoint, headers=headers)

    if not response.is_success:
        response.raise_for_status()
    return response.text if raw else response.json()
