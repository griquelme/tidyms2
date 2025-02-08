import matplotlib.pyplot as plt

from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.io import MSData
from tidyms2.algorithms.raw import make_chromatograms, MakeChromatogramParameters

factory_spec = {
    "data_acquisition": {
        "n_scans": 100,
        "time_resolution": 0.5,
        "int_std": 2.0,
    },
    "adducts": [
        {
            "formula": "[C54H104O6]+",
            "n_isotopologues": 3,
            "abundance": {
                "mean": 500,
                "std": 50,
            },
            "rt": {
                "mean": 10.0,
                "std": 1.0,
                "width": 4.0,
            },
            "noise": {
                "snr": 200,
            }
        },
        {
            "formula": "[C27H40O2]+",
            "n_isotopologues": 2,
            "abundance": {
                "mean": 200,
                "std": 50,
                "prevalence": 0.7,
            },
            "rt": {"mean": 20.0},
        },
        {
            "formula": "[C24H26O12]+",
            "n_isotopologues": 5,
            "rt": {"mean": 30.0},
        },
    ],
}

simulated_sample_factory = SimulatedLCMSSampleFactory(**factory_spec)
sample = simulated_sample_factory(id="my_sample")
ms_data = MSData(sample)

mz = ms_data.get_spectrum(10).mz[0] # get the m/z value from the first feature


params = MakeChromatogramParameters(mz=[mz])
chrom = make_chromatograms(ms_data, params)[0]

fig, ax = plt.subplots()

ax.plot(chrom.time, chrom.int)
ax.set_xlabel("Rt [s]")
ax.set_ylabel("Intensity [au]")
