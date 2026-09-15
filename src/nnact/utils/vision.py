"""Vision datasets used by activation examples."""

from typing import Any

from torch.utils.data import Dataset


class Cifar10Samples(Dataset[dict[str, Any]]):
    """CIFAR-10 images prepared as model keyword inputs.

    The optional ``datasets`` and ``torchvision`` packages are imported only
    when the dataset is instantiated.
    """

    def __init__(self, n: int = 512) -> None:
        from datasets import load_dataset
        from torchvision import transforms

        self._dataset = load_dataset("uoft-cs/cifar10", split="test")
        self._count = min(n, len(self._dataset))
        self.class_names = self._dataset.features["label"].names
        self._transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._dataset[idx]
        return {"x": self._transform(row["img"].convert("RGB"))}
