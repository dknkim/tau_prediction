import monai
import numpy as np
import torch


BRAAK_TARGETS = (
    "Braak1", "Braak2", "Braak3", "Braak4", "Braak5", "Braak6", "Braak1to4", "Braak5_6"
)


class Loader:
    """Load paired MRI volumes and eight regional tau regression targets."""

    def __init__(
        self,
        container,
        scans,
        transforms,
        tracer,
        *,
        classification=False,
        histogram_normalize=False,
        brain_mask_centering=False,
        top_percentile=None,
        encode_dataset=False,
        clip=False,
        use_status=False,
    ):
        if any((classification, histogram_normalize, brain_mask_centering,
                top_percentile is not None, encode_dataset, clip, use_status)):
            raise ValueError("Loader supports the paired MRI regression configuration.")
        self.container = container
        self.scans = scans
        self.transforms = transforms
        self.indicies = self._make_indices(tracer)

    def _make_indices(self, tracer):
        required_scans = {scan.lower() for scan in self.scans}
        visits = {}
        for index, entry in enumerate(self.container):
            value = entry.Braak1to4
            if value is None or isinstance(value, str) or np.isnan(value):
                continue
            if not entry.tracer or tracer not in entry.tracer:
                continue
            sequences = {instance.sequence_type.lower() for instance in entry}
            if not required_scans.issubset(sequences):
                continue
            dates = visits.setdefault(entry.patient_id, {})
            dates.setdefault(entry[0].date, []).append(index)
        return [
            (index, patient, date)
            for patient, dates in visits.items()
            for date, indices in dates.items()
            for index in indices
        ]

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, index):
        entry_index, _, _ = self.indicies[index]
        entry = self.container[entry_index]
        images = []
        for scan in self.scans:
            candidates = [
                instance for instance in entry
                if instance.sequence_type == scan and "brain_mask" not in instance.image_path
            ]
            image = np.random.choice(candidates).open()
            volume = image.get_fdata()
            volume[np.isnan(volume)] = 0
            images.append(volume)

        inputs = monai.data.MetaTensor(np.stack(images), affine=image.affine)
        values = [getattr(entry, name) for name in BRAAK_TARGETS]
        target = torch.tensor(
            [value * 100 if value is not None else 0.0 for value in values],
            dtype=torch.float32,
        )
        return self.transforms(inputs), target
