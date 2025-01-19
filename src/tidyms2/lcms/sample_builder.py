"""Utilities to build LC-MS samples."""

import csv
import pathlib
from typing import Any

import yaml

from ..core.exceptions import SampleNotFound
from ..core.models import Sample, SampleMetadata


def create_csv_sample_template(
    pattern: str,
    template_path: pathlib.Path | str,
    loc: pathlib.Path | None = None,
    **kwargs,
) -> None:
    """Create a csv file template will all files in the data location that matches the provided pattern.

    :param pattern: A glob expression to select files. For example, passing ``"*.mzML"`` will select
        all files with mzML extension in the provided location.
    :param template_path: path where the template file
    :param loc: The directory where the glob expression will be evaluated. If not provided, it will
        be set to the current working directory.
    :param kwargs: default values passed to sample parameters, e.g. `start_time`, `ms_level`, `reader`, etc...
    :raises SampleNotFound: If no samples are found at the provided location.

    """
    loc = loc or pathlib.Path.cwd()

    samples: list[dict[str, Any]] = list()

    for order, file in enumerate(sorted(loc.glob(pattern), key=lambda x: x.name), start=1):
        sample = Sample(id=file.stem, path=file, **kwargs)
        sample.meta.order = order

        d = sample.model_dump()
        meta_d = d.pop("meta")
        d.update(meta_d)
        samples.append(d)

    if not samples:
        raise SampleNotFound(f"No samples at {loc} match the pattern `{pattern}`.")

    if isinstance(template_path, str):
        template_path = pathlib.Path(template_path)

    with template_path.open("wt") as f_out:
        headers = list(samples[0])

        writer = csv.DictWriter(f_out, headers)
        writer.writeheader()
        writer.writerows(samples)


def create_yaml_sample_template(
    pattern: str,
    template_path: pathlib.Path | str,
    loc: pathlib.Path | None = None,
    **kwargs,
) -> None:
    """Create a csv file template will all files in the data location that matches the provided pattern.

    :param pattern: A glob expression to select files. For example, passing ``"*.mzML"`` will select
        all files with mzML extension in the provided location.
    :param template_path: path where the template file
    :param loc: The directory where the glob expression will be evaluated. If not provided, it will
        be set to the current working directory.
    :param kwargs: default values passed to sample parameters, e.g. `start_time`, `ms_level`, `reader`, etc...
    :raises SampleNotFound: If no samples are found at the provided location.

    """
    loc = loc or pathlib.Path.cwd()

    samples: list[dict[str, Any]] = list()

    for order, file in enumerate(sorted(loc.glob(pattern), key=lambda x: x.name), start=1):
        sample = Sample(id=file.stem, path=file, **kwargs)
        sample.meta.order = order
        samples.append(sample.model_dump(mode="json"))

    if not samples:
        raise SampleNotFound(f"No samples at {loc} match the pattern `{pattern}`.")

    if isinstance(template_path, str):
        template_path = pathlib.Path(template_path)

    with template_path.open("wt") as file:
        yaml.dump(samples, file, yaml.Dumper)


def read_samples_from_yaml_template(template_path: pathlib.Path | str) -> list[Sample]:
    """Create samples using a YAML template file created with :py:func:`create_yaml_sample_template`.

    :param template_path: the path to the template file.

    """
    if isinstance(template_path, str):
        template_path = pathlib.Path(template_path)

    with template_path.open("rt") as file:
        samples = yaml.load(file, yaml.Loader)

    assert isinstance(samples, list), "Expected YAML file to contain a list of sample models."

    return [Sample(**d) for d in samples]


def read_samples_from_csv_template(template_path: pathlib.Path | str) -> list[Sample]:
    """Create samples using a CSV template file created with :py:func:`create_csv_sample_template`.

    :param template_path: the path to the template file.

    """
    if isinstance(template_path, str):
        template_path = pathlib.Path(template_path)

    with template_path.open("rt") as fin:
        dialect = csv.Sniffer().sniff(fin.read(2048))
        fin.seek(0)
        reader = csv.DictReader(fin, dialect=dialect)

        sample_fields = list(Sample.model_fields)
        samples: list[Sample] = list()
        for row in reader:
            row = {k: v for k, v in row.items() if v}  # ignore empty records in csv
            meta = SampleMetadata(**{k: v for k, v in row.items() if k not in sample_fields})  # type: ignore
            sample = Sample(meta=meta, **{k: v for k, v in row.items() if k in sample_fields})  # type: ignore
            samples.append(sample)
    return samples
