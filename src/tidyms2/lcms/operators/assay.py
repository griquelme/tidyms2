"""LC-MS assay operators."""

from typing import TYPE_CHECKING

import pydantic

from ...annotation.correspondence import FeatureCorrespondenceParameters, match_features
from ...core.enums import MSInstrument, Polarity, SeparationMode
from ...core.models import AnnotationPatch, MZTrace
from ...core.operators.assay import AnnotationPatcher
from ...core.storage import AssayStorage
from ..models import Peak

if TYPE_CHECKING:
    from typing import assert_never


class LCFeatureMatcher(AnnotationPatcher, FeatureCorrespondenceParameters):
    """Perform feature correspondence on LC-MS datasets using a cluster-based approach.

    Features are initially grouped by m/z and Rt similarity using DBSCAN. In a second step, these
    clusters are further processed using a GMM approach, obtaining clusters where each sample
    contributes with only one sample.

    See the :ref:`algorithms-correspondence` guide for a detailed description of the algorithm.

    """

    mz_tolerance: pydantic.PositiveFloat = 0.01
    """m/z tolerance used to group close features. Sets the `eps` parameter in the DBSCAN algorithm."""

    rt_tolerance: pydantic.PositiveFloat = 3.0
    """Rt tolerance in seconds used to group close features. Sets the `eps` parameter in the DBSCAN algorithm."""

    def compute_patches(self, data: AssayStorage[MZTrace, Peak]) -> list[AnnotationPatch]:
        """Compute annotation patches for feature matching."""
        tolerance = {"mz": self.mz_tolerance, "rt": self.rt_tolerance}
        descriptors = data.fetch_descriptors(descriptors=list(tolerance))
        annotations = data.fetch_annotations()
        samples = data.list_samples()
        return match_features(descriptors, annotations, samples, tolerance, self)

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        """Set the processor default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        op = cls()

        match instrument:
            case MSInstrument.QTOF:
                op.mz_tolerance = 0.01
            case MSInstrument.ORBITRAP:
                op.mz_tolerance = 0.005
            case _ as never:
                assert_never(never)

        match separation:
            case SeparationMode.HPLC:
                op.rt_tolerance = 10.0
            case SeparationMode.UPLC:
                op.rt_tolerance = 5.0
            case SeparationMode.DART:
                raise NotImplementedError
            case _ as never:
                assert_never(never)
        return op
