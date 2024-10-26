import os
import pathlib

import pytest

from tidyms2.io.datasets import download_dataset


@pytest.fixture(scope="session")
def data_dir():
    tmp_data_path = pathlib.Path(__file__).parent.parent / "data"
    if not tmp_data_path.exists():
        tmp_data_path.mkdir(exist_ok=True)
        gitignore = tmp_data_path / ".gitignore"
        gitignore.write_text("*\n")
    return tmp_data_path


@pytest.fixture(scope="session")
def raw_data_dir(data_dir: pathlib.Path) -> pathlib.Path:
    res = data_dir / "raw"

    if not res.exists():
        res.mkdir()
        dataset = "test-raw-data"
        download_dataset(dataset, res, token=os.getenv("GH_PAT"))
    return res
