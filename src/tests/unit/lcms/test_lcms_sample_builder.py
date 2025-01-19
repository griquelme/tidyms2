import pathlib

import pytest

from tidyms2.core.exceptions import SampleNotFound
from tidyms2.lcms import sample_builder

from ..helpers import create_sample


class TestBuildFromYAML:
    def test_template_from_location_without_files_raise_error(self, tmp_path: pathlib.Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        template_file = tmp_path / "template.yaml"
        with pytest.raises(SampleNotFound):
            sample_builder.create_yaml_sample_template("*", template_file, data_dir)

    def test_create_template_and_read_returns_equal_samples(self, tmp_path: pathlib.Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        expected = list()
        for k in range(1, 6):
            s = create_sample(data_dir, k)
            expected.append(s)

        template_file = tmp_path / "template.yaml"

        sample_builder.create_yaml_sample_template("*", template_file, data_dir)

        actual = sample_builder.read_samples_from_yaml_template(template_file)
        assert actual == expected


class TestBuildFromCSV:
    def test_template_from_location_without_files_raise_error(self, tmp_path: pathlib.Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        template_file = tmp_path / "template.csv"
        with pytest.raises(SampleNotFound):
            sample_builder.create_csv_sample_template("*", template_file, data_dir)

    def test_create_template_and_read_returns_equal_samples(self, tmp_path: pathlib.Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        expected = list()
        for k in range(1, 6):  # start at order=1 so it matches order create by the builder function
            s = create_sample(data_dir, k)
            expected.append(s)

        template_file = tmp_path / "template.csv"

        sample_builder.create_csv_sample_template("*", template_file, data_dir)

        actual = sample_builder.read_samples_from_csv_template(template_file)
        assert actual == expected
