from pathlib import Path

import matplotlib.pyplot as plt

from tidyms2.core.models import Sample
from tidyms2.simulation.lcms import SimulatedLCMSAdductSpec, simulate_data_matrix

formula_list = ["[C54H104O6]+", "[C27H40O2]+", "[C24H26O12]+"]
rt_list = [10.0, 20.0, 30.0]
abundances = [1000.0, 2000.0, 3000.0]
adducts = list()
for rt, abundance, formula in zip(rt_list, abundances, formula_list):
    adduct = SimulatedLCMSAdductSpec(
        formula=formula,
        n_isotopologues=2,
        abundance={"mean": abundance, "std": 20.0}, # type: ignore
        response={"max_sensitivity_loss": 0.5, "sensitivity_decay": 0.25, "interbatch_variation": 0.5}, # type: ignore
    )
    adducts.append(adduct)


samples = list()
for k in range(40):
    id_ = f"sample-{k}"
    path = Path.cwd() / f"{id_}.mzML"
    s = Sample(id=id_, path=path)
    s.meta.order = k
    s.meta.batch = k // 10
    samples.append(s)

data = simulate_data_matrix(adducts, samples)

col = data.get_columns(0)[0]
order = [s.meta.order for s in data.list_samples()]

fig, ax = plt.subplots()

ax.plot(order, col.data, color="tab:grey")
ax.set_title(f"mz={col.feature.mz:.4f}, Rt={col.feature.descriptors["rt"]:.1f} s")
ax.fill_between([0, 9], 100, 500, alpha=0.25, label="batch=1")
ax.fill_between([10, 19], 100, 500, alpha=0.25, label="batch=2")
ax.fill_between([20, 29], 100, 500, alpha=0.25, label="batch=3")
ax.fill_between([30, 39], 100, 500, alpha=0.25, label="batch=4")
ax.legend()


ax.set_xlabel("Run order")
ax.set_ylabel("Intensity [au]")
ax.set_ylim(0, 600.0)
