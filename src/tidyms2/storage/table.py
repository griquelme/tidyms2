"""Container class for assay features descriptors and annotations."""

from typing import Iterable
from uuid import UUID

from ..core.exceptions import FeatureGroupNotFound, FeatureNotFound, SampleNotFound
from ..core.models import Annotation, AnnotationPatch, DescriptorPatch, FeatureGroup


class FeatureTable:
    """Store Feature annotations and descriptors in long, i.e., one feature record per row, format."""

    def __init__(self):
        self._descriptors: dict[str, list[float]] = dict()
        self._annotations: list[Annotation] = list()
        self._fill_values: dict[str, dict[int, float]] = dict()
        self._feature_groups: dict[int, FeatureGroup] = dict()

        self._sample_to_features: dict[str, set[UUID]] = dict()
        self._group_to_features: dict[int, set[UUID]] = dict()

        # maintains feature index in the snapshot for fast access
        self._feature_to_index: dict[UUID, int] = dict()
        self._current_index = 0

    def add_descriptors(self, descriptors: list[dict[str, float]], annotations: list[Annotation]):
        """Add annotations and descriptors from a sample."""
        for descriptor in descriptors:
            for name, value in descriptor.items():
                descriptor_list = self._descriptors.setdefault(name, list())
                descriptor_list.append(value)
        self._annotations.extend(annotations)

        for k, ann in enumerate(annotations, start=self._current_index):
            sample_set = self._sample_to_features.setdefault(ann.sample_id, set())
            sample_set.add(ann.id)
            if ann.group > -1:
                group_set = self._group_to_features.setdefault(ann.group, set())
                group_set.add(ann.id)

            self._feature_to_index[ann.id] = k

        self._current_index += len(descriptors)

    def add_feature_groups(self, *feature_groups: FeatureGroup) -> None:
        """Add feature groups to the snapshot."""
        for group in feature_groups:
            self._feature_groups[group.group] = group

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups from the snapshot."""
        return [x.model_copy(deep=True) for x in self._feature_groups.values()]

    def has_feature_group(self, group: int) -> bool:
        """Check if a group with the provided id is stored in the assay."""
        return group in self._feature_groups

    def fetch_annotations(self, sample_id: str | None = None, copy: bool = False) -> list[Annotation]:
        """Create a list feature annotations.

        :param sample_id: If provided, only include annotations from this sample
        """
        if sample_id is None:
            result = [x for x in self._annotations]
        elif sample_id not in self._sample_to_features:
            raise SampleNotFound(sample_id)
        else:
            indices = [self._feature_to_index[x] for x in self._sample_to_features[sample_id]]
            result = [self._annotations[x] for x in indices]

        if copy:
            result = [x.model_copy() for x in result]

        return result

    def fetch_descriptors(
        self, descriptors: Iterable[str] | None = None, sample_id: str | None = None, copy: bool = False
    ) -> dict[str, list[float]]:
        """Fetch descriptors from the snapshot."""
        if descriptors is None:
            descriptors = list(self._descriptors)
        if sample_id is None and copy:
            result = {x: self._descriptors[x].copy() for x in descriptors}
        elif sample_id is None:
            result = {x: self._descriptors[x] for x in descriptors}
        elif sample_id not in self._sample_to_features:
            raise SampleNotFound(sample_id)
        else:
            indices = [self._feature_to_index[x] for x in self._sample_to_features[sample_id]]
            result = dict()
            for name in descriptors:
                values = self._descriptors[name]
                result[name] = [values[x] for x in indices]
        return result

    def fetch_fill_values(self, copy: bool = False) -> dict[str, dict[int, float]]:
        """Fetch snapshot fill values."""
        if copy:
            return {k: v.copy() for k, v in self._fill_values.items()}
        return self._fill_values

    def add_fill_values(self, fill_values: dict[str, dict[int, float]]) -> None:
        """Add missing values to the snapshot."""
        for sample_id, features_fill in fill_values.items():
            for feature_group, value in features_fill.items():
                sample_fill = self._fill_values.setdefault(sample_id, dict())
                sample_fill[feature_group] = value

    def get_sample_id(self, feature_id: UUID) -> str:
        """Retrieve the sample id of a feature."""
        index = self._feature_to_index[feature_id]
        return self._annotations[index].sample_id

    def get_ids_by_group(self, group: int) -> list[UUID]:
        """Retrieve all feature ids associated with a feature group."""
        if group not in self._group_to_features:
            raise FeatureGroupNotFound(group)
        return [x for x in self._group_to_features[group]]

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature is in the snapshot."""
        return feature_id in self._feature_to_index

    def patch_annotation(self, *patches: AnnotationPatch) -> None:
        """Apply patches to annotations."""
        for p in patches:
            if not self.has_feature(p.id):
                raise FeatureNotFound(p.id)

        for p in patches:
            index = self._feature_to_index[p.id]
            ann = self._annotations[index]
            setattr(ann, p.field, p.value)
            self._update_group_to_features(p)

    def list_feature_groups(self) -> list[int]:
        """List all feature groups stored in the assay."""
        return list(self._feature_groups)

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Apply patches to descriptors."""
        for p in patches:
            if not self.has_feature(p.id):
                raise FeatureNotFound(p.id)

        for p in patches:
            index = self._feature_to_index[p.id]
            self._descriptors[p.descriptor][index] = p.value

    def _update_group_to_features(self, patch: AnnotationPatch):
        """Update the group to feature id mapping when the group annotation is update."""
        if patch.field == "group" and patch.value > -1:
            group_set = self._group_to_features.setdefault(patch.value, set())
            group_set.add(patch.id)
