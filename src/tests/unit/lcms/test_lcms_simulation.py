import numpy as np
import pytest

from tidyms2.core.models import MSSpectrum
from tidyms2.io import MSData
from tidyms2.lcms import simulation


@pytest.fixture
def factory_with_grid(lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
    lcms_sample_factory.config.grid = simulation.MZGridSpecification()
    return lcms_sample_factory


class TestSimulatedLCMSSample:
    def test_make_grid_no_spec_ok(self, lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
        sample = lcms_sample_factory(id="sample")
        assert sample.meta is not None
        simulated_sample_spec = simulation.SimulatedLCMSSample(**sample.meta.model_dump())
        grid = simulated_sample_spec.make_grid()
        assert grid.size == len(simulated_sample_spec.features)
        assert np.all(np.diff(grid) > 0.0)

    def test_make_grid_with_spec(self, factory_with_grid: simulation.SimulatedLCMSSampleFactory):
        sample = factory_with_grid(id="sample")
        assert sample.meta is not None
        simulated_sample_spec = simulation.SimulatedLCMSSample(**sample.meta.model_dump())
        grid = simulated_sample_spec.make_grid()
        assert simulated_sample_spec.config.grid is not None
        assert grid.size == simulated_sample_spec.config.grid.size
        assert np.all(np.diff(grid) > 0.0)

    def test_make_grid_no_features_return_empty_array(self):
        factory = simulation.SimulatedLCMSSampleFactory()
        sample = factory(id="sample")
        assert sample.meta is not None
        simulated_sample_spec = simulation.SimulatedLCMSSample(**sample.meta.model_dump())
        grid = simulated_sample_spec.make_grid()
        assert grid.size == 0


class TestSimulatedMSData:
    @pytest.fixture(scope="class")
    def data(self, lcms_sample_factory: simulation.SimulatedLCMSSampleFactory):
        sample = lcms_sample_factory(id="sample")
        return MSData(sample)

    def test_get_spectrum(self, data: MSData):
        index = 0
        sp = data.get_spectrum(index)
        assert isinstance(sp, MSSpectrum)
        assert sp.index == index

    def test_get_spectrum_negative_index_raises_value_error(self, data: MSData):
        with pytest.raises(ValueError):
            data.get_spectrum(-1)

    def test_get_spectrum_index_greater_than_size_raises_error(self, data: MSData):
        with pytest.raises(ValueError):
            index = 100000000
            data.get_spectrum(index)

    def test_multiple_calls_to_get_spectrum_retrieves_the_same_instance(self, data: MSData):
        index = 10
        sp1 = data.get_spectrum(index)
        sp2 = data.get_spectrum(index)
        assert sp1 is sp2
