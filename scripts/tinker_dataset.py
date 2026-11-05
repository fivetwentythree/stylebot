"""A Cookbook ChatDatasetBuilder that keeps the final incomplete batch."""

import math
import random
from pathlib import Path

import chz
import tinker
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

from experiment import verify_prepared
from review_batch import read_jsonl


def encode_datum(datum):
    return {
        "model_input": datum.model_input.model_dump(mode="json"),
        "loss_fn_inputs": {
            name: {"data": tensor.data, "dtype": tensor.dtype, "shape": tensor.shape}
            for name, tensor in datum.loss_fn_inputs.items()
        },
    }


def decode_datum(row):
    from tinker.types import ModelInput, TensorData

    return tinker.Datum(
        model_input=ModelInput.model_validate(row["model_input"]),
        loss_fn_inputs={
            name: TensorData(**tensor) for name, tensor in row["loss_fn_inputs"].items()
        },
    )


class CompleteBatches(SupervisedDataset):
    def __init__(self, datums, batch_size, seed=0):
        self.datums = list(datums)
        self.batch_size = batch_size
        self.seed = seed
        self.order = list(range(len(self.datums)))
        if not self.datums or batch_size <= 0:
            raise ValueError("A dataset needs data and a positive batch size")

    def __len__(self):
        return math.ceil(len(self.datums) / self.batch_size)

    def get_batch(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ids = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        return [self.datums[i] for i in ids]

    def set_epoch(self, seed=0):
        self.order = list(range(len(self.datums)))
        random.Random(self.seed + seed).shuffle(self.order)


@chz.chz
class FrozenChatBuilder(ChatDatasetBuilder):
    prepared_dir: str
    shuffle_seed: int = 0

    def __call__(self):
        root = Path(self.prepared_dir)
        verify_prepared(root, token_report=True)

        def load(split):
            return [
                decode_datum(r)
                for r in read_jsonl(root / "tokenized" / f"{split}.jsonl")
            ]

        training, validation = load("train"), load("validation")
        return (
            CompleteBatches(training, self.common_config.batch_size, self.shuffle_seed),
            CompleteBatches(validation, len(validation)),
        )
