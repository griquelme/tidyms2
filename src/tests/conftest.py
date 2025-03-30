import os
import pathlib

import pytest
from numpy.random import seed

from tidyms2.io.datasets import download_dataset
from tidyms2.simulation import lcms


@pytest.fixture(scope="session", autouse=True)
def random_seed():
    seed(1234)
    return


@pytest.fixture(scope="session")
def data_dir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def raw_data_dir(data_dir: pathlib.Path) -> pathlib.Path:
    raw_data_dir = data_dir / "raw"

    if not raw_data_dir.exists():
        raw_data_dir.mkdir()
        dataset = "test-raw-data"
        download_dataset(dataset, raw_data_dir, token=os.getenv("GH_PAT"))

    gitignore = raw_data_dir / ".gitignore"
    gitignore.write_text("*\n")
    return raw_data_dir


@pytest.fixture(scope="module")
def lcms_adducts():
    formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
    rt_list = [10.0, 20.0, 30.0]
    abundances = [1000.0, 2000.0, 3000.0]
    adduct_list = list()
    for rt, abundance, formula in zip(rt_list, abundances, formula_list):
        adduct = lcms.SimulatedLCMSAdductSpec(
            formula=formula,
            rt={"mean": rt},  # type: ignore
            abundance={"mean": abundance},  # type: ignore
            n_isotopologues=2,
        )
        adduct_list.append(adduct)
    return adduct_list


@pytest.fixture(scope="module")
def lcms_sample_factory(lcms_adducts):
    config = lcms.DataAcquisitionSpec(n_scans=40, int_std=0.0)
    return lcms.SimulatedLCMSSampleFactory(data_acquisition=config, adducts=lcms_adducts)
