from sedna.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDecodingDatasetProcessor")
class SpeculativeDecodingDatasetProcessor:
    def __init__(self, **kwargs):
        sample_size = kwargs.get("sample_size")
        self.sample_size = int(sample_size) if sample_size is not None else None
        warmup_samples = kwargs.get("warmup_samples")
        self.warmup_samples = max(0, int(warmup_samples)) if warmup_samples is not None else 0

    def __call__(self, dataset):
        dataset_name = getattr(dataset, "dataset_name", "default")
        processed = []
        for index, (x, y) in enumerate(zip(dataset.x, dataset.y)):
            processed.append(
                {
                    "request_id": f"request-{index:03d}",
                    "query": x,
                    "gold": y,
                    "task_name": dataset_name,
                    "sample_index": index,
                }
            )

        if self.sample_size is not None and self.sample_size > 0:
            processed = processed[: self.sample_size]
            dataset.y = dataset.y[: self.sample_size]

        effective_warmup = min(self.warmup_samples, len(processed))
        for index, item in enumerate(processed):
            item["sample_index"] = index
            item["warmup_samples"] = effective_warmup
            item["is_warmup"] = index < effective_warmup

        dataset.x = processed
        return dataset
