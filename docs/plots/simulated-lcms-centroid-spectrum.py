import matplotlib.pyplot as plt

from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.io import MSData

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

sp = ms_data.get_spectrum(10)

fig, ax = plt.subplots()
ax.stem(sp.mz, sp.int, basefmt=" ", markerfmt="")
ax.set_xlabel("m/z")
ax.set_ylabel("Intensity [au]")
ax.set_xlim(845, 855)
ax.set_ylim(0, 50)
