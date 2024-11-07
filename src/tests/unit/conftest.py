import os
import pathlib

import pytest

from tidyms2.io.datasets import download_dataset
from tidyms2.lcms import simulation


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


@pytest.fixture(scope="module")
def lcms_sample_factory():
    config = simulation.SimulatedLCMSDataConfiguration(min_signal_intensity=1.0, n_scans=40, amp_noise=0.0)
    formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
    rt_list = [10.0, 20.0, 30.0]
    int_list = [100.0, 200.0, 300.0]
    adduct_list = list()
    for rt, spint, formula in zip(rt_list, int_list, formula_list):
        adduct = simulation.SimulatedLCMSAdductSpec(
            formula=formula, rt_mean=rt, base_intensity=spint, n_isotopologues=2
        )
        adduct_list.append(adduct)
    return simulation.SimulatedLCMSSampleFactory(config=config, adducts=adduct_list)
