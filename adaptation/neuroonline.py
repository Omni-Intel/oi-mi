"""NeuroOnline training mechanics adapted to the project's EEG decoders.

Every observed sample gets one time-masked and one frequency-masked view. All
three views receive supervised classification loss, the augmented
representations are aligned with the original representation, and the backbone,
context modulator, and classifier are optimized together.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.factory import (
    BaseModelAdapter,
    TorchModelAdapter,
    atomic_torch_save,
    enable_mc_dropout,
    split_train_validation_indices,
)

LOGGER = logging.getLogger(__name__)
NEUROONLINE_TRAINING_MECHANICS_VERSION = 2


class ClassCollapseError(RuntimeError):
    """Raised when training never produces a checkpoint covering every class."""

    def __init__(self, message: str, metrics: dict[str, float]) -> None:
        super().__init__(message)
        self.metrics = metrics


@dataclass(frozen=True, slots=True)
class NeuroOnlineConfig:
    """Hyperparameters matching the released motor-imagery pipeline."""

    enabled: bool = False
    learning_rate: float = 1e-6
    update_batch_size: int = 16
    epochs: int = 3
    update_stride: int = 64
    history_threshold: int = 64
    recent_samples: int = 320
    weight_decay: float = 5e-2
    mask_ratio: float = 0.7
    consistency_weight: float = 0.1
    label_smoothing: float = 0.1
    prompt_count: int = 32
    random_seed: int = 42
    offline_epochs: int = 50
    offline_patience: int = 50
    offline_batch_size: int = 16
    offline_learning_rate: float = 1e-4

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "NeuroOnlineConfig":
        root = payload or {}
        strategy = str(root.get("strategy", "periodic_head")).strip().lower()
        data = root.get("neuroonline", {}) or {}
        enabled = bool(root.get("enabled", False)) and strategy == "neuroonline"
        return cls(
            enabled=enabled,
            learning_rate=max(float(data.get("learning_rate", 1e-6)), 1e-9),
            update_batch_size=max(int(data.get("update_batch_size", 16)), 1),
            epochs=max(int(data.get("epochs", 3)), 1),
            update_stride=max(int(data.get("update_stride", 64)), 1),
            history_threshold=max(int(data.get("history_threshold", 64)), 1),
            recent_samples=max(int(data.get("recent_samples", 320)), 1),
            weight_decay=max(float(data.get("weight_decay", 5e-2)), 0.0),
            mask_ratio=min(max(float(data.get("mask_ratio", 0.7)), 0.0), 1.0),
            consistency_weight=max(float(data.get("consistency_weight", 0.1)), 0.0),
            label_smoothing=min(max(float(data.get("label_smoothing", 0.1)), 0.0), 1.0),
            prompt_count=max(int(data.get("prompt_count", 32)), 1),
            random_seed=int(data.get("random_seed", 42)),
            offline_epochs=max(int(data.get("offline_epochs", 50)), 1),
            offline_patience=max(int(data.get("offline_patience", 50)), 1),
            offline_batch_size=max(int(data.get("offline_batch_size", 16)), 1),
            offline_learning_rate=max(float(data.get("offline_learning_rate", 1e-4)), 1e-9),
        )


class ContextAwareRepresentationModulator(nn.Module):
    """Released CRM design generalized to a model classifier's input shape."""

    def __init__(self, *, token_count: int, embedding_dim: int, prompt_count: int = 32) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.embedding_dim = int(embedding_dim)
        self.prompt_count = int(prompt_count)
        self.subject_codes = nn.Parameter(
            torch.randn(self.prompt_count, self.token_count, self.embedding_dim) * 0.01
        )
        self.router = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.prompt_count),
        )
        self.norm_q = nn.LayerNorm(self.embedding_dim)
        self.norm_kv = nn.LayerNorm(self.embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=_attention_heads(self.embedding_dim),
            dropout=0.1,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 2 * self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
        )
        self.alpha_head = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.beta_head = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"CRM expects [batch, tokens, embedding], got {tuple(tokens.shape)}")
        if tokens.shape[1:] != (self.token_count, self.embedding_dim):
            raise ValueError(
                "CRM representation shape changed from "
                f"({self.token_count}, {self.embedding_dim}) to {tuple(tokens.shape[1:])}"
            )
        pooled = tokens.mean(dim=1)
        routing = F.softmax(self.router(pooled), dim=-1)
        prompt = (routing[:, :, None, None] * self.subject_codes[None, :, :, :]).sum(dim=1)
        attention, _ = self.attn(self.norm_q(prompt), self.norm_kv(tokens), self.norm_kv(tokens))
        hidden = tokens + attention
        hidden = hidden + self.mlp(self.norm2(hidden))
        alpha = 1.0 + self.gate_alpha * self.alpha_head(hidden)
        beta = self.gate_beta * self.beta_head(hidden)
        return alpha, beta


class NeuroOnlineModelAdapter(BaseModelAdapter):
    """Add CRM and the NeuroOnline objective to a PyTorch decoder."""

    def __init__(
        self,
        base: TorchModelAdapter,
        *,
        config: NeuroOnlineConfig,
        state_path: Path | None = None,
    ) -> None:
        self.base = base
        self.model_name = base.model_name
        self._device = base._device
        self._classifier = _find_classifier(base.model)
        self._modulator: ContextAwareRepresentationModulator | None = None
        self._feature_shape: tuple[int, ...] | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._update_generator = torch.Generator().manual_seed(config.random_seed)
        self._state_path = state_path
        self._pending_state = _load_neuroonline_state(state_path, self._device)
        self.config = _checkpoint_config(config, self._pending_state)
        self._update_generator.manual_seed(self.config.random_seed)

    def _prepare_training(self, example: torch.Tensor) -> ContextAwareRepresentationModulator:
        """Initialize CRM lazily and make the complete NeuroOnline stack trainable."""

        self._ensure_modulator(example.to(self._device))
        assert self._modulator is not None
        for module in (self.base.model, self._modulator):
            for parameter in module.parameters():
                parameter.requires_grad = True
        return self._modulator

    def _view_loader(
        self,
        original: torch.Tensor | np.ndarray,
        time_masked: torch.Tensor | np.ndarray,
        frequency_masked: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        *,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(original, dtype=torch.float32),
            torch.as_tensor(time_masked, dtype=torch.float32),
            torch.as_tensor(frequency_masked, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=max(int(batch_size), 1),
            shuffle=shuffle,
        )

    def _training_objective(
        self,
        original: torch.Tensor,
        time_masked: torch.Tensor,
        frequency_masked: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, original_representation = self._forward_adapted(original)
        time_logits, time_representation = self._forward_adapted(time_masked)
        frequency_logits, frequency_representation = self._forward_adapted(
            frequency_masked
        )
        classification = (
            criterion(logits, labels)
            + criterion(time_logits, labels)
            + criterion(frequency_logits, labels)
        )
        consistency = (
            F.mse_loss(time_representation, original_representation)
            + F.mse_loss(frequency_representation, original_representation)
        ) / 2.0
        return (
            classification + self.config.consistency_weight * consistency,
            classification,
            consistency,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        clip_classifier_gradients: bool = False,
    ) -> dict[str, float]:
        assert self._modulator is not None
        self.base.model.train()
        self._modulator.train()
        totals = {"loss": 0.0, "classification_loss": 0.0, "consistency_loss": 0.0}
        batches = 0
        for batch in loader:
            original, time_masked, frequency_masked, labels = (
                value.to(self._device) for value in batch
            )
            loss, classification, consistency = self._training_objective(
                original,
                time_masked,
                frequency_masked,
                labels,
                criterion,
            )
            optimizer.zero_grad()
            loss.backward()
            if clip_classifier_gradients:
                trainable_parameters = [
                    parameter
                    for parameter in self._classifier.parameters()
                    if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            totals["loss"] += float(loss.item())
            totals["classification_loss"] += float(classification.item())
            totals["consistency_loss"] += float(consistency.item())
            batches += 1
        return {name: value / max(batches, 1) for name, value in totals.items()}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool = False,
        groups: np.ndarray | None = None,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        del epochs, batch_size, learning_rate, patience, head_only
        train_indices, validation_indices = split_train_validation_indices(
            y,
            groups=groups,
            random_state=self.config.random_seed,
        )
        return self.fit_with_split(
            X,
            y,
            train_indices=train_indices,
            validation_indices=validation_indices,
            groups=groups,
            progress_callback=progress_callback,
        )

    def fit_with_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        groups: np.ndarray | None = None,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Fit with an explicit split so every search candidate sees identical trials."""

        from sklearn.metrics import (
            balanced_accuracy_score,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            log_loss,
        )

        train_indices = np.asarray(train_indices, dtype=np.int64)
        validation_indices = np.asarray(validation_indices, dtype=np.int64)
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError("NeuroOnline training and validation splits must both be non-empty.")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ValueError("NeuroOnline training and validation indices overlap.")
        trial_groups = None if groups is None else np.asarray(groups, dtype=np.int64)
        if trial_groups is not None and trial_groups.shape != np.asarray(y).shape:
            raise ValueError("NeuroOnline trial groups must match the labels shape.")

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        generator = torch.Generator().manual_seed(self.config.random_seed)
        all_inputs = torch.as_tensor(X, dtype=torch.float32)
        time_views = _time_mask(all_inputs, self.config.mask_ratio, generator)
        frequency_views = _frequency_mask(all_inputs, self.config.mask_ratio, generator)
        modulator = self._prepare_training(all_inputs[:1])
        loader = self._view_loader(
            all_inputs[train_indices],
            time_views[train_indices],
            frequency_views[train_indices],
            torch.as_tensor(y[train_indices], dtype=torch.long),
            batch_size=self.config.offline_batch_size,
        )
        optimizer = torch.optim.AdamW(
            list(self.base.model.parameters()) + list(modulator.parameters()),
            lr=self.config.offline_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.config.offline_epochs * len(loader), 1),
            eta_min=1e-6,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing).to(self._device)
        best_balanced_accuracy = float("-inf")
        best_kappa = float("-inf")
        best_accuracy = 0.0
        best_macro_f1 = 0.0
        best_worst_class_accuracy = 0.0
        best_trial_balanced_accuracy = float("-inf")
        best_trial_kappa = float("-inf")
        best_trial_accuracy = 0.0
        best_trial_macro_f1 = 0.0
        best_trial_worst_class_accuracy = 0.0
        best_loss = float("inf")
        best_model_state: dict[str, torch.Tensor] | None = None
        best_modulator_state: dict[str, torch.Tensor] | None = None
        best_score: tuple[float, ...] | None = None
        latest_validation_metrics: dict[str, float] = {}
        stagnant_epochs = 0
        epochs_completed = 0
        for epoch_index in range(self.config.offline_epochs):
            metrics = self._train_epoch(
                loader,
                optimizer,
                criterion,
                scheduler=scheduler,
                clip_classifier_gradients=True,
            )
            probabilities = self.predict_proba(X[validation_indices])
            predictions = probabilities.argmax(axis=1)
            truth = y[validation_indices]
            kappa = float(cohen_kappa_score(truth, predictions))
            if not np.isfinite(kappa):
                kappa = -1.0
            accuracy = float(np.mean(predictions == truth))
            validation_loss = float(
                log_loss(
                    truth,
                    probabilities,
                    labels=np.arange(probabilities.shape[1]),
                )
            )
            balanced_accuracy = float(balanced_accuracy_score(truth, predictions))
            macro_f1 = float(f1_score(truth, predictions, average="macro", zero_division=0))
            matrix = confusion_matrix(
                truth,
                predictions,
                labels=np.arange(int(np.max(y)) + 1),
            )
            class_totals = matrix.sum(axis=1)
            class_accuracies = np.divide(
                np.diag(matrix),
                class_totals,
                out=np.zeros_like(class_totals, dtype=np.float64),
                where=class_totals > 0,
            )
            worst_class_accuracy = float(np.min(class_accuracies[class_totals > 0]))
            if trial_groups is not None:
                trial_truth, trial_probabilities = _aggregate_trial_probabilities(
                    truth,
                    probabilities,
                    trial_groups[validation_indices],
                )
                trial_predictions = trial_probabilities.argmax(axis=1)
            else:
                trial_truth = truth
                trial_predictions = predictions
            trial_accuracy = float(np.mean(trial_predictions == trial_truth))
            trial_balanced_accuracy = float(
                balanced_accuracy_score(trial_truth, trial_predictions)
            )
            trial_kappa = float(cohen_kappa_score(trial_truth, trial_predictions))
            if not np.isfinite(trial_kappa):
                trial_kappa = -1.0
            trial_macro_f1 = float(
                f1_score(
                    trial_truth,
                    trial_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            trial_matrix = confusion_matrix(
                trial_truth,
                trial_predictions,
                labels=np.arange(int(np.max(y)) + 1),
            )
            trial_totals = trial_matrix.sum(axis=1)
            trial_class_accuracies = np.divide(
                np.diag(trial_matrix),
                trial_totals,
                out=np.zeros_like(trial_totals, dtype=np.float64),
                where=trial_totals > 0,
            )
            trial_worst_class_accuracy = float(
                np.min(trial_class_accuracies[trial_totals > 0])
            )
            epochs_completed = epoch_index + 1
            latest_validation_metrics = {
                "val_loss": validation_loss,
                "val_acc": accuracy,
                "val_balanced_accuracy": balanced_accuracy,
                "val_kappa": kappa,
                "val_macro_f1": macro_f1,
                "val_worst_class_accuracy": worst_class_accuracy,
                "val_trial_acc": trial_accuracy,
                "val_trial_balanced_accuracy": trial_balanced_accuracy,
                "val_trial_kappa": trial_kappa,
                "val_trial_macro_f1": trial_macro_f1,
                "val_trial_worst_class_accuracy": trial_worst_class_accuracy,
                "epochs_completed": float(epochs_completed),
            }
            if progress_callback is not None:
                progress_callback(
                    epochs_completed,
                    self.config.offline_epochs,
                    {
                        **metrics,
                        **latest_validation_metrics,
                    },
                )
            noncollapsed = (
                worst_class_accuracy > 0.0
                and trial_worst_class_accuracy > 0.0
            )
            score = (
                trial_worst_class_accuracy,
                worst_class_accuracy,
                trial_balanced_accuracy,
                trial_kappa,
                trial_macro_f1,
                balanced_accuracy,
                -validation_loss,
            )
            if noncollapsed and (best_score is None or score > best_score):
                best_balanced_accuracy = balanced_accuracy
                best_kappa = kappa
                best_accuracy = accuracy
                best_macro_f1 = macro_f1
                best_worst_class_accuracy = worst_class_accuracy
                best_trial_balanced_accuracy = trial_balanced_accuracy
                best_trial_kappa = trial_kappa
                best_trial_accuracy = trial_accuracy
                best_trial_macro_f1 = trial_macro_f1
                best_trial_worst_class_accuracy = trial_worst_class_accuracy
                best_loss = validation_loss
                best_score = score
                best_model_state = _copy_state_dict(self.base.model)
                best_modulator_state = _copy_state_dict(modulator)
                stagnant_epochs = 0
            elif best_score is not None:
                stagnant_epochs += 1
                if stagnant_epochs >= self.config.offline_patience:
                    LOGGER.info(
                        "NeuroOnline offline calibration stopped after %s epochs "
                        "(no validation-score improvement for %s epochs).",
                        epochs_completed,
                        self.config.offline_patience,
                    )
                    break
        if best_model_state is None or best_modulator_state is None:
            raise ClassCollapseError(
                "NeuroOnline training produced no checkpoint with non-zero "
                "window and trial recall for every validation class.",
                latest_validation_metrics,
            )
        self.base.model.load_state_dict(best_model_state)
        self._modulator.load_state_dict(best_modulator_state)
        self._optimizer = None
        return {
            "val_loss": best_loss,
            "val_acc": best_accuracy,
            "val_balanced_accuracy": best_balanced_accuracy,
            "val_kappa": best_kappa,
            "val_macro_f1": best_macro_f1,
            "val_worst_class_accuracy": best_worst_class_accuracy,
            "val_trial_acc": best_trial_accuracy,
            "val_trial_balanced_accuracy": best_trial_balanced_accuracy,
            "val_trial_kappa": best_trial_kappa,
            "val_trial_macro_f1": best_trial_macro_f1,
            "val_trial_worst_class_accuracy": best_trial_worst_class_accuracy,
            "epochs_completed": float(epochs_completed),
        }

    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        inputs = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        passes = max(int(mc_dropout_passes), 1)
        outputs: list[np.ndarray] = []
        self._ensure_modulator(inputs[:1])
        assert self._modulator is not None
        if passes > 1:
            enable_mc_dropout(self.base.model)
            enable_mc_dropout(self._modulator)
        else:
            self.base.model.eval()
            self._modulator.eval()
        for _ in range(passes):
            with torch.no_grad():
                logits, _ = self._forward_adapted(inputs)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.mean(np.stack(outputs, axis=0), axis=0)

    def save(self, path: Path) -> None:
        if self._modulator is None and self._pending_state is None:
            self.base.save(path)
            return
        state = self._pending_state
        if state is None:
            assert self._modulator is not None
            state = {
                "training_mechanics_version": NEUROONLINE_TRAINING_MECHANICS_VERSION,
                "feature_shape": self._feature_shape,
                "config": asdict(self.config),
                "modulator": self._modulator.state_dict(),
            }
        atomic_torch_save(
            {
                "checkpoint_format": "neuroonline_bundle_v1",
                "model_state_dict": self.base.model.state_dict(),
                "neuroonline": state,
            },
            path,
        )
        sidecar = _sidecar_path(path)
        atomic_torch_save(state, sidecar)

    def load(self, path: Path) -> None:
        self.base.load(path)
        self._modulator = None
        self._feature_shape = None
        self._optimizer = None
        self._state_path = path
        self._pending_state = _load_neuroonline_state(self._state_path, self._device)
        self.config = _checkpoint_config(self.config, self._pending_state)
        self._update_generator.manual_seed(self.config.random_seed)

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> dict[str, float]:
        time_view = _time_mask(torch.as_tensor(X), self.config.mask_ratio, None).numpy()
        freq_view = _frequency_mask(torch.as_tensor(X), self.config.mask_ratio, None).numpy()
        return self.neuroonline_update(
            X,
            time_view,
            freq_view,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

    def neuroonline_update(
        self,
        X: np.ndarray,
        X_time: np.ndarray,
        X_freq: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        if X.size == 0 or y.size == 0:
            return {"updated": 0.0, "loss": 0.0}
        inputs = torch.as_tensor(X, dtype=torch.float32)
        modulator = self._prepare_training(inputs[:1])
        lr = self.config.learning_rate if learning_rate is None else float(learning_rate)
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                list(self.base.model.parameters()) + list(modulator.parameters()),
                lr=lr,
                weight_decay=self.config.weight_decay,
            )
        permutation = torch.randperm(
            inputs.shape[0],
            generator=self._update_generator,
        )
        loader = self._view_loader(
            inputs[permutation],
            torch.as_tensor(X_time)[permutation],
            torch.as_tensor(X_freq)[permutation],
            torch.as_tensor(y)[permutation],
            batch_size=max(int(batch_size or self.config.update_batch_size), 1),
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing).to(self._device)
        metrics = {"loss": 0.0, "classification_loss": 0.0, "consistency_loss": 0.0}
        for _ in range(max(int(epochs or self.config.epochs), 1)):
            metrics = self._train_epoch(
                loader,
                self._optimizer,
                criterion,
                clip_classifier_gradients=False,
            )
        return {
            "updated": float(X.shape[0]),
            **metrics,
            "gate_alpha": float(modulator.gate_alpha.detach().cpu().item()),
            "gate_beta": float(modulator.gate_beta.detach().cpu().item()),
        }

    def _ensure_modulator(self, example: torch.Tensor) -> None:
        if self._modulator is not None:
            return
        self.base.model.eval()
        with torch.no_grad():
            features = self._extract_features(example)
        tokens, feature_shape = _features_to_tokens(features)
        self._feature_shape = feature_shape
        self._modulator = ContextAwareRepresentationModulator(
            token_count=tokens.shape[1],
            embedding_dim=tokens.shape[2],
            prompt_count=self.config.prompt_count,
        ).to(self._device)
        if self._pending_state is not None:
            expected = tuple(self._pending_state.get("feature_shape") or ())
            if expected and expected != self._feature_shape:
                raise ValueError(
                    f"Saved NeuroOnline feature shape {expected} does not match {self._feature_shape}"
                )
            self._modulator.load_state_dict(self._pending_state["modulator"])
            self._pending_state = None

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured.append(args[0])

        handle = self._classifier.register_forward_pre_hook(capture)
        try:
            self.base.model(inputs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Could not capture the classifier input for NeuroOnline CRM")
        return captured[-1]

    def _forward_adapted(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_features(inputs)
        tokens, feature_shape = _features_to_tokens(features)
        if self._feature_shape != feature_shape:
            raise ValueError(f"Classifier input shape changed from {self._feature_shape} to {feature_shape}")
        assert self._modulator is not None
        alpha, beta = self._modulator(tokens)
        adapted_tokens = tokens * alpha + beta
        adapted_features = _tokens_to_features(adapted_tokens, feature_shape)
        logits = _normalize_logits(self._classifier(adapted_features))
        return logits, adapted_tokens


class NeuroOnlineStreamAdapter:
    """Causal predict-then-update coordinator driven by labeled sample count."""

    def __init__(
        self,
        *,
        config: NeuroOnlineConfig,
        update_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]],
        save_callback: Callable[[], None] | None = None,
        completion_callback: Callable[[dict[str, Any]], None] | None = None,
        n_classes: int = 3,
    ) -> None:
        self.config = config
        self._update_callback = update_callback
        self._save_callback = save_callback
        self._completion_callback = completion_callback
        self._original: deque[np.ndarray] = deque(maxlen=config.recent_samples)
        self._time: deque[np.ndarray] = deque(maxlen=config.recent_samples)
        self._frequency: deque[np.ndarray] = deque(maxlen=config.recent_samples)
        self._labels: deque[int] = deque(maxlen=config.recent_samples)
        self._window_ids: deque[int] = deque(maxlen=config.recent_samples)
        self._event_ids: deque[str] = deque(maxlen=config.recent_samples)
        self._model_revisions: deque[int] = deque(maxlen=config.recent_samples)
        self._generator = torch.Generator().manual_seed(config.random_seed)
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._pending_update = False
        self._n_classes = max(int(n_classes), 1)
        self._confusion = np.zeros((self._n_classes, self._n_classes), dtype=np.int64)
        self._operational_confusion = np.zeros(
            (self._n_classes, self._n_classes),
            dtype=np.int64,
        )
        self._operational_abstentions = 0
        self._seen = 0
        self._updates = 0
        self._state = "collecting"
        self._last_result: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._accepted_event_ids: set[str] = set()
        self._last_window_end_monotonic: float | None = None
        self._duplicate_windows_rejected = 0
        self._stale_windows_rejected = 0

    def add_window(
        self,
        window: np.ndarray,
        label: int,
        *,
        predicted_label: int | None = None,
        operational_predicted_label: int | None = None,
        probabilities: np.ndarray | None = None,
        event_id: str = "",
        model_revision: int = 0,
        window_end_monotonic: float | None = None,
    ) -> bool:
        del probabilities
        sample = torch.as_tensor(np.asarray(window, dtype=np.float32)).unsqueeze(0)
        with self._lock:
            if self._closed:
                return False
            stable_event_id = str(event_id).strip()
            if stable_event_id and stable_event_id in self._accepted_event_ids:
                self._duplicate_windows_rejected += 1
                return False
            end_timestamp = (
                None
                if window_end_monotonic is None
                else float(window_end_monotonic)
            )
            if end_timestamp is not None:
                if not np.isfinite(end_timestamp):
                    self._stale_windows_rejected += 1
                    return False
                if (
                    self._last_window_end_monotonic is not None
                    and end_timestamp <= self._last_window_end_monotonic
                ):
                    self._stale_windows_rejected += 1
                    return False
            time_view = _time_mask(
                sample,
                self.config.mask_ratio,
                self._generator,
            ).squeeze(0).numpy()
            frequency_view = _frequency_mask(
                sample,
                self.config.mask_ratio,
                self._generator,
            ).squeeze(0).numpy()
            self._original.append(sample.squeeze(0).numpy().copy())
            self._time.append(time_view)
            self._frequency.append(frequency_view)
            self._labels.append(int(label))
            self._seen += 1
            self._window_ids.append(self._seen)
            self._event_ids.append(stable_event_id)
            self._model_revisions.append(int(model_revision))
            if stable_event_id:
                self._accepted_event_ids.add(stable_event_id)
            if end_timestamp is not None:
                self._last_window_end_monotonic = end_timestamp
            if (
                predicted_label is not None
                and 0 <= int(label) < self._n_classes
                and 0 <= int(predicted_label) < self._n_classes
            ):
                self._confusion[int(label), int(predicted_label)] += 1
            if 0 <= int(label) < self._n_classes:
                if (
                    operational_predicted_label is not None
                    and 0 <= int(operational_predicted_label) < self._n_classes
                ):
                    self._operational_confusion[
                        int(label),
                        int(operational_predicted_label),
                    ] += 1
                else:
                    self._operational_abstentions += 1
            should_update = (
                self._seen >= self.config.history_threshold
                and self._seen % self.config.update_stride == 0
            )
            if not should_update:
                if self._worker is None:
                    self._state = "collecting"
                return True
            if self._worker is not None:
                self._pending_update = True
                self._state = "training"
                return True
            self._start_update_locked(self._snapshot_locked())
            return True

    def close(self, *, timeout_sec: float = 60.0) -> None:
        with self._lock:
            self._closed = True
            self._pending_update = False
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(float(timeout_sec), 0.0))
        if self._save_callback is not None and self._updates > 0:
            self._save_callback()

    def wait_for_idle(self, *, timeout_sec: float = 60.0) -> bool:
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while time.monotonic() <= deadline:
            with self._lock:
                worker = self._worker
                idle = worker is None and not self._pending_update
            if idle:
                return True
            if worker is not None:
                worker.join(timeout=min(max(deadline - time.monotonic(), 0.0), 0.05))
            else:
                time.sleep(0.01)
        return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        next_step = max(self.config.history_threshold, self._seen + 1)
        if self._seen >= self.config.history_threshold:
            remainder = self._seen % self.config.update_stride
            next_step = self._seen + (self.config.update_stride - remainder if remainder else self.config.update_stride)
        if self._seen < self.config.history_threshold:
            phase_start = 0
            phase_target = self.config.history_threshold
        else:
            phase_target = next_step
            phase_start = phase_target - self.config.update_stride
        progress = (self._seen - phase_start) / max(phase_target - phase_start, 1)
        counts = np.bincount(np.asarray(self._labels, dtype=np.int64), minlength=self._n_classes)
        return {
            "enabled": self.config.enabled,
            "strategy": "neuroonline",
            "config": asdict(self.config),
            "state": self._state,
            "seen_labeled_windows": self._seen,
            "buffered_windows": len(self._labels),
            "update_count": self._updates,
            "training_in_background": self._worker is not None,
            "pending_update": self._pending_update,
            "duplicate_windows_rejected": self._duplicate_windows_rejected,
            "stale_windows_rejected": self._stale_windows_rejected,
            "next_update_step": next_step,
            "samples_until_update": max(next_step - self._seen, 0),
            "progress": min(max(float(progress), 0.0), 1.0),
            "class_counts": {str(index): int(counts[index]) for index in range(self._n_classes)},
            "prequential": self._prequential_metrics_locked(),
            "operational_prequential": self._operational_metrics_locked(),
            "update_history": list(self._history),
            "last_result": self._last_result,
        }

    def _snapshot_locked(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        labels = np.asarray(self._labels, dtype=np.int64)
        window_ids = list(self._window_ids)
        return (
            np.stack(self._original).astype(np.float32),
            np.stack(self._time).astype(np.float32),
            np.stack(self._frequency).astype(np.float32),
            labels,
            {
                "trigger_seen_labeled_windows": self._seen,
                "snapshot_first_window_id": int(window_ids[0]),
                "snapshot_last_window_id": int(window_ids[-1]),
                "snapshot_samples": int(labels.size),
                "snapshot_class_counts": np.bincount(
                    labels,
                    minlength=self._n_classes,
                ).tolist(),
                "snapshot_first_event_id": self._event_ids[0],
                "snapshot_last_event_id": self._event_ids[-1],
                "snapshot_first_model_revision": int(self._model_revisions[0]),
                "snapshot_last_model_revision": int(self._model_revisions[-1]),
                "trigger_timestamp_unix": time.time(),
                "trigger_timestamp_monotonic": time.monotonic(),
            },
        )

    def _start_update_locked(
        self,
        snapshot: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, Any],
        ],
    ) -> None:
        self._state = "training"
        self._worker = threading.Thread(
            target=self._run_update,
            args=snapshot,
            name=f"neuroonline-update-{self._updates + 1}",
            daemon=True,
        )
        self._worker.start()

    def _run_update(
        self,
        original: np.ndarray,
        time_masked: np.ndarray,
        frequency_masked: np.ndarray,
        labels: np.ndarray,
        snapshot_metadata: dict[str, Any],
    ) -> None:
        started_at = time.perf_counter()
        started_unix = time.time()
        try:
            result = dict(self._update_callback(original, time_masked, frequency_masked, labels))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("NeuroOnline background update failed")
            result = {"updated": 0.0, "error": str(exc)}
        result["duration_sec"] = float(time.perf_counter() - started_at)
        result.update(snapshot_metadata)
        result["training_started_unix"] = started_unix
        result["training_completed_unix"] = time.time()
        succeeded = not result.get("error") and float(result.get("updated", 0.0)) > 0

        if succeeded and self._save_callback is not None:
            try:
                self._save_callback()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to persist NeuroOnline update")
                result["save_error"] = str(exc)

        with self._lock:
            if succeeded:
                self._updates += 1
            self._last_result = result
            prequential = self._prequential_metrics_locked()
            history_item: dict[str, Any] = {
                "update": self._updates,
                "seen_labeled_windows": int(
                    result.get("trigger_seen_labeled_windows", self._seen)
                ),
                "trigger_seen_labeled_windows": int(
                    result.get("trigger_seen_labeled_windows", self._seen)
                ),
                "snapshot_first_window_id": int(
                    result.get("snapshot_first_window_id", 0)
                ),
                "snapshot_last_window_id": int(
                    result.get("snapshot_last_window_id", 0)
                ),
                "snapshot_samples": int(result.get("snapshot_samples", labels.size)),
                "snapshot_class_counts": list(
                    result.get("snapshot_class_counts", [])
                ),
                "snapshot_first_event_id": str(
                    result.get("snapshot_first_event_id", "")
                ),
                "snapshot_last_event_id": str(
                    result.get("snapshot_last_event_id", "")
                ),
                "base_model_revision": int(result.get("base_model_revision", 0)),
                "model_revision": int(result.get("model_revision", 0)),
                "trigger_timestamp_unix": float(
                    result.get("trigger_timestamp_unix", 0.0)
                ),
                "training_started_unix": float(
                    result.get("training_started_unix", 0.0)
                ),
                "training_completed_unix": float(
                    result.get("training_completed_unix", 0.0)
                ),
                "loss": float(result.get("loss", 0.0)),
                "classification_loss": float(result.get("classification_loss", 0.0)),
                "consistency_loss": float(result.get("consistency_loss", 0.0)),
                "gate_alpha": float(result.get("gate_alpha", 0.0)),
                "gate_beta": float(result.get("gate_beta", 0.0)),
                "duration_sec": float(result["duration_sec"]),
                "prequential_accuracy": float(prequential["accuracy"]),
                "prequential_balanced_accuracy": float(prequential["balanced_accuracy"]),
            }
            if result.get("error"):
                history_item["error"] = str(result["error"])
            self._history.append(history_item)
            self._worker = None
            if self._pending_update and not self._closed:
                self._pending_update = False
                self._start_update_locked(self._snapshot_locked())
            else:
                self._state = "collecting" if not self._closed else "closed"
        if self._completion_callback is not None:
            try:
                self._completion_callback(dict(result))
            except Exception:  # noqa: BLE001
                LOGGER.exception("NeuroOnline completion callback failed")

    def _prequential_metrics_locked(self) -> dict[str, Any]:
        support = self._confusion.sum(axis=1)
        correct = int(np.trace(self._confusion))
        evaluated = int(self._confusion.sum())
        per_class = np.divide(
            np.diag(self._confusion),
            support,
            out=np.zeros(self._n_classes, dtype=np.float64),
            where=support > 0,
        )
        balanced = float(per_class.mean())
        return {
            "evaluated_windows": evaluated,
            "correct_windows": correct,
            "accuracy": float(correct / evaluated) if evaluated else 0.0,
            "balanced_accuracy": balanced,
            "all_classes_observed": bool(np.all(support > 0)),
            "per_class_accuracy": {
                str(index): float(per_class[index]) for index in range(self._n_classes)
            },
            "confusion_matrix": self._confusion.tolist(),
        }

    def _operational_metrics_locked(self) -> dict[str, Any]:
        support = self._confusion.sum(axis=1)
        correct = int(np.trace(self._operational_confusion))
        evaluated = int(self._confusion.sum())
        issued = int(self._operational_confusion.sum())
        per_class = np.divide(
            np.diag(self._operational_confusion),
            support,
            out=np.zeros(self._n_classes, dtype=np.float64),
            where=support > 0,
        )
        return {
            "evaluated_windows": evaluated,
            "issued_commands": issued,
            "abstained_windows": self._operational_abstentions,
            "coverage": float(issued / evaluated) if evaluated else 0.0,
            "accuracy_with_abstention_as_error": (
                float(correct / evaluated) if evaluated else 0.0
            ),
            "selective_accuracy": float(correct / issued) if issued else 0.0,
            "balanced_accuracy": float(per_class.mean()),
            "per_class_recall": {
                str(index): float(per_class[index])
                for index in range(self._n_classes)
            },
            "confusion_matrix": self._operational_confusion.tolist(),
        }


def _find_classifier(model: nn.Module) -> nn.Module:
    for name in ("classifier", "final_layer"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, nn.Module):
            return candidate
    raise ValueError(
        f"Model {type(model).__name__} does not expose a classifier/final_layer for NeuroOnline CRM"
    )


def _copy_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _features_to_tokens(features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(features.shape[1:])
    if features.ndim == 2:
        return features.unsqueeze(1), shape
    permutation = [0, *range(2, features.ndim), 1]
    tokens = features.permute(*permutation).reshape(features.shape[0], -1, features.shape[1])
    return tokens, shape


def _tokens_to_features(tokens: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 1:
        return tokens.squeeze(1)
    spatial = shape[1:]
    arranged = tokens.reshape(tokens.shape[0], *spatial, shape[0])
    permutation = [0, len(shape), *range(1, len(shape))]
    return arranged.permute(*permutation)


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    while logits.ndim > 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError(f"NeuroOnline classifier must produce [batch, classes], got {tuple(logits.shape)}")
    return logits


def _attention_heads(embedding_dim: int) -> int:
    for heads in (4, 2, 1):
        if embedding_dim % heads == 0:
            return heads
    return 1


def _time_mask(
    inputs: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if inputs.ndim < 2:
        raise ValueError("Time masking expects a tensor with a time dimension.")
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return inputs.clone()
    mask = torch.rand(
        inputs.shape,
        generator=generator,
        device=inputs.device,
    ) < ratio
    return inputs.masked_fill(mask, 0.0)


def _frequency_mask(
    inputs: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if inputs.ndim < 2:
        raise ValueError("Frequency masking expects a tensor with a time dimension.")
    ratio = float(np.clip(ratio, 0.0, 1.0))
    spectrum = torch.fft.rfft(inputs, dim=-1)
    if ratio <= 0.0:
        return inputs.clone()
    mask = torch.rand(
        spectrum.shape,
        generator=generator,
        device=inputs.device,
    ) < ratio
    masked = spectrum.masked_fill(mask, 0.0 + 0.0j)
    return torch.fft.irfft(masked, n=inputs.shape[-1], dim=-1)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.neuroonline.pt")


def _load_neuroonline_state(path: Path | None, device: torch.device) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.name.endswith(".neuroonline.pt") and path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            embedded = checkpoint.get("neuroonline")
            if isinstance(embedded, dict):
                return _validate_neuroonline_state(embedded)
    sidecar = path if path.name.endswith(".neuroonline.pt") else _sidecar_path(path)
    if not sidecar.exists():
        return None
    payload = torch.load(sidecar, map_location=device, weights_only=True)
    return _validate_neuroonline_state(payload)


def _validate_neuroonline_state(payload: dict[str, Any]) -> dict[str, Any]:
    saved_version = int(payload.get("training_mechanics_version", 1))
    if saved_version != NEUROONLINE_TRAINING_MECHANICS_VERSION:
        LOGGER.warning(
            "Loading NeuroOnline sidecar trained with mechanics version %s; "
            "current training uses version %s. Retrain before deployment.",
            saved_version,
            NEUROONLINE_TRAINING_MECHANICS_VERSION,
        )
    return payload


def _checkpoint_config(
    fallback: NeuroOnlineConfig,
    payload: dict[str, Any] | None,
) -> NeuroOnlineConfig:
    """Restore model-coupled settings selected during calibration."""

    saved = (payload or {}).get("config", {}) or {}
    supported = {
        field_name: saved[field_name]
        for field_name in (
            "mask_ratio",
            "consistency_weight",
            "weight_decay",
            "label_smoothing",
            "prompt_count",
            "random_seed",
        )
        if field_name in saved
    }
    return replace(fallback, **supported) if supported else fallback


def _aggregate_trial_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trial_labels: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for group in np.unique(groups):
        mask = groups == group
        unique_labels = np.unique(labels[mask])
        if unique_labels.size != 1:
            raise ValueError(
                f"Validation trial {int(group)} contains multiple labels: "
                f"{unique_labels.tolist()}."
            )
        trial_labels.append(int(unique_labels[0]))
        trial_probabilities.append(np.mean(probabilities[mask], axis=0))
    return (
        np.asarray(trial_labels, dtype=np.int64),
        np.stack(trial_probabilities, axis=0),
    )
