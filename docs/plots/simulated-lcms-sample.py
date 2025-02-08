import matplotlib.pyplot as plt

from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.io import MSData
from tidyms2.algorithms.raw import make_chromatograms, MakeChromatogramParameters

adducts_spec = [
    {
        "formula": "[C54H104O6]+",
        "rt": {"mean": 10.0},
    },
    {
        "formula": "[C27H40O2]+",
        "rt": {"mean": 20.0},
    },
    {
        "formula": "[C24H26O12]+",
        "rt": {"mean": 30.0},
    },
]

simulated_sample_factory = SimulatedLCMSSampleFactory(adducts=adducts_spec) # type: ignore
sample = simulated_sample_factory(id="my_sample")
ms_data = MSData(sample)

mz = ms_data.get_spectrum(10).mz[0] # get the m/z value from the first feature


params = MakeChromatogramParameters(mz=[mz])
chrom = make_chromatograms(ms_data, params)[0]

fig, ax = plt.subplots()

ax.plot(chrom.time, chrom.int)
ax.set_xlabel("Rt [s]")
ax.set_ylabel("Intensity [au]")
