"""Single-fidelity active learning with control over how much data each retraining uses

Supports two training policies which share the same retraining schedule:

- *Full*: retrain on every record in the database (ExaMol's default behavior)
- *Incremental*: retrain only on records added since the last retraining.
  Use with a scorer that warm-starts from the previous model (e.g., ``NFPScorer(retrain_from_scratch=False)``)
"""
import json
import random
from pathlib import Path
from time import perf_counter, time

import numpy as np
from colmena.thinker import agent, event_responder

from .single import SingleStepThinker
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe


class IncrementalThinker(SingleStepThinker):
    """A :class:`~examol.steer.single.SingleStepThinker` which retrains after a fixed number of new results,
    optionally using only the new results for training

    Also records the training set used in each round (``training-rounds.json``) and the
    ensemble predictions for the whole search space after each round (``predictions/``).

    Args:
        retrain_every: Minimum number of newly-completed molecules before retraining
        infer_every: Number of newly-completed molecules after which to re-run inference with the current
            models, re-ranking the search space without retraining. Set to ``None`` to only run inference after training
        incremental: Whether to train only on records not used in a previous training round
        replay_size: When training incrementally, how many previously-used records to mix back in to each
            round, sampled at random. Keeps the per-round cost constant while showing the model older data
        replay_seed: Seed for choosing the replay sample
    """

    def __init__(self, *args, retrain_every: int = 64, infer_every: int | None = None, incremental: bool = False,
                 replay_size: int = 0, replay_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain_every = retrain_every
        self.infer_every = infer_every
        self.incremental = incremental
        self.replay_size = replay_size
        self._replay_rng = random.Random(replay_seed)
        self._num_replayed: int = 0  # Records replayed in the most recent round
        self._completed_at_last_inference: int = 0

        self._trained_keys: set[str] = set()  # Records used in (or available to) a previous training round
        self._pending_keys: set[str] = set()  # Records available to the current training round
        self._round: int = 0  # Index of the most recent training round
        self._inference_pass: int = 0  # Number of inference passes so far (may be more than one per training round)
        self._training_active: bool = False  # Whether models are being trained right now
        self._dispatched: set[str] = set()  # SMILES of molecules which have been sent to simulation
        self._predictions_dir = Path(self.run_dir) / 'predictions'
        self._predictions_dir.mkdir(parents=True, exist_ok=True)
        self.last_val_loss: float | None = None  # Best validation loss from the most recent training round, averaged over the ensemble

    def _compute_val_loss(self) -> float | None:
        """Average the best (minimum) validation loss each ensemble member reached this round, across recipes and models"""
        per_model = [min(h['val_loss']) for histories in self.last_train_history.values() for h in histories if 'val_loss' in h]
        return float(np.mean(per_model)) if per_model else None

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        full_set = super()._get_training_set(recipe)
        self._pending_keys.update(r.key for r in full_set)
        if self.incremental and len(self._trained_keys) > 0:
            new = [r for r in full_set if r.key not in self._trained_keys]
            if self.replay_size > 0:
                old = [r for r in full_set if r.key in self._trained_keys]
                replayed = self._replay_rng.sample(old, min(self.replay_size, len(old)))
                self._num_replayed = len(replayed)
                new += replayed
            return new
        return full_set

    def get_additional_training_information(self, train_set: list[MoleculeRecord], recipe: PropertyRecipe) -> dict[str, object]:
        output = super().get_additional_training_information(train_set, recipe)
        if self.incremental:
            # Fit the output scaling only on the first round. Afterward, each round sees only a (biased) batch of new
            #  data, and re-fitting the scale to that batch would shift predictions for every molecule.
            # Exception: a scorer that always retrains from scratch has no warm-started weights to preserve a
            #  previous scale from, so it must be refit every round.
            output['reset_scale'] = len(self._trained_keys) == 0 or getattr(self.scorer, 'retrain_from_scratch', False)
        return output

    def _count_new(self) -> int:
        return min(
            sum(1 for r in super(IncrementalThinker, self)._get_training_set(recipe) if r.key not in self._trained_keys)
            for recipe in self.recipes
        )

    @event_responder(event_name='start_training')
    def retrain(self):
        """Retrain all models once enough new data are available"""

        # Same guards as the parent class, plus waiting for enough new data
        if self.start_inference.is_set():
            self.logger.info('Inference is still ongoing. Will not retrain yet')
            return
        train_size = min(self.count_training_size(r) for r in self.recipes)
        if train_size < self.solution.minimum_training_size:
            self.logger.info(f'Too few entries to train. Waiting for {self.solution.minimum_training_size}. Have {train_size}')
            return
        n_new = self._count_new()
        if len(self._trained_keys) > 0 and n_new < self.retrain_every:
            self.logger.info(f'Only {n_new} new records since last training. Waiting for {self.retrain_every}')
            return

        # Run the training using the parent's (undecorated) implementation
        self._pending_keys = set()
        start_wall, start_time = time(), perf_counter()
        self._round += 1
        self.logger.info(f'Starting training round {self._round}. {n_new} new records, incremental={self.incremental}')
        self._training_active = True
        try:
            SingleStepThinker.retrain.__wrapped__(self)
        finally:
            self._training_active = False
        self._completed_at_last_inference = self.completed  # Training ends by running inference
        self.last_val_loss = self._compute_val_loss()

        # Log the training round
        n_used = len(self._pending_keys - self._trained_keys) if self.incremental and len(self._trained_keys) > 0 else len(self._pending_keys)
        with (Path(self.run_dir) / 'training-rounds.json').open('a') as fp:
            print(json.dumps({
                'round': self._round,
                'start': start_wall,
                'runtime': perf_counter() - start_time,
                'num_available': len(self._pending_keys),
                'num_trained_on': n_used,
                'num_new': n_new,
                'num_replayed': self._num_replayed,
                'incremental': self.incremental,
                'val_loss': self.last_val_loss,
                'trained_keys': (sorted(self._pending_keys - self._trained_keys) if self.incremental and len(self._trained_keys) > 0
                                 else sorted(self._pending_keys)),
            }), file=fp)
        self._trained_keys.update(self._pending_keys)

    @agent()
    def trigger_inference(self):
        """Re-rank the search space with the current models between training rounds"""
        if self.infer_every is None:
            return
        while not self.done.is_set():
            self.done.wait(1)
            ready = (self._round > 0 and not self._training_active and not self.start_inference.is_set()
                     and self.completed - self._completed_at_last_inference >= self.infer_every)
            if not ready:
                continue
            if self._count_new() >= self.retrain_every:
                continue  # A training round is due, and ExaMol will not retrain while inference is running
            self._completed_at_last_inference = self.completed
            self.logger.info(f'Re-running inference with the current models after {self.completed} molecules')
            for recipe_id, models in enumerate(self.models):
                for model_id in range(len(models)):
                    self.ready_models.put((recipe_id, model_id))
            self.start_inference.set()

    def _get_next_tasks(self):
        # Record the string as it appears in the search space (the caller holds the task queue lock)
        self._dispatched.add(self.task_queue[0][0])
        return super()._get_next_tasks()

    def _filter_inference_results(self, chunk_id: int, chunk_smiles: list[str], inference_results: np.ndarray) -> tuple[list[str], np.ndarray]:
        # Save the ensemble predictions (recipe x molecule x model) for later analysis of model quality
        if chunk_id == 0:
            self._inference_pass += 1
        np.savez_compressed(
            self._predictions_dir / f'pass-{self._inference_pass:04d}-round-{self._round:04d}-chunk-{chunk_id:03d}.npz',
            smiles=np.array(chunk_smiles), predictions=inference_results.astype(np.float32), time=time(),
        )

        # Do not re-select molecules which have already been sent to simulation
        keep = [i for i, s in enumerate(chunk_smiles) if s not in self._dispatched]
        return [chunk_smiles[i] for i in keep], inference_results[:, keep, :]
