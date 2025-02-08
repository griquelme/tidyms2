import matplotlib.pyplot as plt

from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.io import MSData

factory_spec = {
    "data_acquisition": {
        "grid": {"low": 200.0, "high": 300.0, "size": 100000},
        "mz_width": 0.005,
    },
    "adducts": [
        {
            "formula": "[C54H104O6]3+",
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
    ]
}

simulated_sample_factory = SimulatedLCMSSampleFactory(**factory_spec) # type: ignore
sample = simulated_sample_factory(id="my_sample")
ms_data = MSData(sample)

sp = ms_data.get_spectrum(10)

fig, ax = plt.subplots()
ax.plot(sp.mz, sp.int)
ax.set_xlabel("m/z")
ax.set_ylabel("Intensity [au]")
ax.set_xlim(282, 284)
ax.set_ylim(0, 600)
