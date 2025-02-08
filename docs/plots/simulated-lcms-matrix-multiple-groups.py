from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tidyms2.core.models import Sample
from tidyms2.simulation.lcms import SimulatedLCMSAdductSpec, simulate_data_matrix

formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
rt_list = [10.0, 20.0, 30.0]
abundances = [1000.0, 2000.0, 3000.0]
adducts = list()
for rt, abundance, formula in zip(rt_list, abundances, formula_list):
    adduct = SimulatedLCMSAdductSpec(
        formula=formula,
        abundance={
            "group-a": {"mean": abundance, "std": 50.0},
            "group-b": {"mean": abundance * 2, "std": 50.0}
        },  # type: ignore
    )
    adducts.append(adduct)


samples = list()
for k in range(10):
    id_ = f"sample-{k}"
    path = Path.cwd() / f"{id_}.mzML"
    s = Sample(id=id_, path=path)
    s.meta.order = k
    s.meta.group = "group-a" if k % 2 else "group-b"
    samples.append(s)

data = simulate_data_matrix(adducts, samples)

col = data.get_columns(0)[0]
order = np.array([s.meta.order for s in data.list_samples()])
group = np.array([s.meta.group for s in data.list_samples()])

fig, ax = plt.subplots()

mask_group_a = group == "group-a"
mask_group_b = group == "group-b"
ax.scatter(order[mask_group_a], col.data[mask_group_a], label="group-a")
ax.scatter(order[mask_group_b], col.data[mask_group_b], label="group-b")
ax.legend()
ax.set_title(f"mz={col.feature.mz:.4f}, Rt={col.feature.descriptors["rt"]:.1f} s")
ax.set_xlabel("Run order")
ax.set_ylabel("Intensity [au]")
ax.set_ylim(0, 1200)
