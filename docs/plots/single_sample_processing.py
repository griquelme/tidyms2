import matplotlib.pyplot as plt
from tidyms2.core.models import MZTrace
from tidyms2.lcms import Peak
from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.storage import OnMemorySampleStorage
from tidyms2.lcms.operators.sample import LCTraceExtractor, LCPeakExtractor


# sample simulation
factory_spec = {
    "config": {
        "n_scans": 40,
        "mz_std": 1.0,
    },
    "adducts": [
        {
            "formula": "[C54H104O6]+",
            "rt": {"mean": 10.0},
            "abundance": {"mean": 1000.0},
            "n_isotopologues": 2,
        },
        {
            "formula": "[C27H40O2]+",
            "rt": {"mean": 20.0},
            "abundance": {"mean": 2000.0},
            "n_isotopologues": 2,
        },
        {
            "formula": "[C24H26O12]+",
            "rt": {"mean": 30.0},
            "abundance": {"mean": 3000.0},
            "n_isotopologues": 2,
        },
    ],
}

simulated_sample_factory = SimulatedLCMSSampleFactory(**factory_spec)
sample = simulated_sample_factory(id="my_sample")


# sample storage creation
sample_data = OnMemorySampleStorage(sample, MZTrace, Peak)

# applying trace extractor and peak extractor
trace_extractor = LCTraceExtractor(id="example_roi_extractor", tolerance=0.005)
peak_extractor = LCPeakExtractor(id="example_peak_extractor")


trace_extractor.apply(sample_data)
peak_extractor.apply(sample_data)


peak = sample_data.list_features()[0]
roi = peak.roi
mz_mean = roi.mz.mean().item()

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(roi.time, roi.spint, label="m/z trace")
ax.fill_between(roi.time[peak.start:peak.end + 1], roi.spint[peak.start:peak.end + 1], alpha=0.25, label="peak")
ax.set_ylabel("Intensity")
ax.set_xlabel("Retention Time")
ax.legend()