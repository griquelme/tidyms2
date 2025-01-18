"""Processor class for sample isotopologue annotation."""

from typing import TYPE_CHECKING, Self

from ..core.dataflow import SampleProcessStatus
from ..core.enums import MSInstrument, Polarity, SeparationMode
from ..core.models import AnnotableFeatureType, RoiType
from ..core.operators.sample import SampleOperator
from ..core.storage import SampleStorage
from .annotation import AnnotatorParameters, annotate, create_annotation_tools

if TYPE_CHECKING:
    from typing import assert_never


class IsotopologueAnnotator(SampleOperator[RoiType, AnnotableFeatureType], AnnotatorParameters):
    """Annotate isotopologues in a sample.

    Groups isotopologue features. Each group is assigned an unique label, and a
    charge state. Each feature in a group is assigned an unique index that
    determines the position in the envelope.

    Annotations are stored to the `annotation` attribute of each feature.
    """

    def _apply_operator(self, data: SampleStorage[RoiType, AnnotableFeatureType]):
        tools = create_annotation_tools(self)
        annotate(data.list_features(), *tools)

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        """Set the annotator default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        op = cls.from_chnops(2000)
        op.max_length = 10

        match polarity:
            case Polarity.POSITIVE:
                op.max_charge = 3
            case Polarity.NEGATIVE:
                op.max_charge = -3
            case _ as never:
                assert_never(never)

        match instrument:
            case MSInstrument.QTOF:
                op.min_M_tol = 0.005
                op.max_M_tol = 0.01
            case MSInstrument.ORBITRAP:
                op.min_M_tol = 0.001
                op.max_M_tol = 0.005
            case _ as never:
                assert_never(never)

        return op

    def get_expected_status_in(self) -> SampleProcessStatus:
        """Get the expected status before isotopologue annotation."""
        return SampleProcessStatus(roi_extracted=True, feature_extracted=True)

    def get_expected_status_out(self) -> SampleProcessStatus:
        """Get the expected status after isotopologue annotation."""
        return SampleProcessStatus(roi_extracted=True, feature_extracted=True, isotopologue_annotated=True)
