from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _repo_root_from_package() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_override(name: str, repo_root: Path, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    data_root: Path
    results_root: Path

    @classmethod
    def default(cls) -> "RepoPaths":
        repo_root = _repo_root_from_package()
        data_root = _resolve_override("RNI_DATA_ROOT", repo_root, repo_root / "data")
        results_root = _resolve_override("RNI_RESULTS_ROOT", repo_root, repo_root / "results")
        return cls(repo_root=repo_root, data_root=data_root, results_root=results_root)

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def raw_images_root(self) -> Path:
        return self.raw_root / "images"

    @property
    def raw_texts_root(self) -> Path:
        return self.raw_root / "texts"

    @property
    def reps_root(self) -> Path:
        return self.data_root / "reps"

    @property
    def cnn_results_root(self) -> Path:
        return self.results_root / "cnns"

    @property
    def cnn_metrics_root(self) -> Path:
        return self.cnn_results_root / "metrics"

    @property
    def cnn_figs_root(self) -> Path:
        return self.cnn_results_root / "figs"

    @property
    def llm_results_root(self) -> Path:
        return self.results_root / "llms"

    @property
    def llm_metrics_root(self) -> Path:
        return self.llm_results_root / "metrics"

    @property
    def llm_figs_root(self) -> Path:
        return self.llm_results_root / "figs"

    @property
    def vit_results_root(self) -> Path:
        return self.results_root / "vits"

    @property
    def vit_metrics_root(self) -> Path:
        return self.vit_results_root / "metrics"

    @property
    def vit_figs_root(self) -> Path:
        return self.vit_results_root / "figs"
