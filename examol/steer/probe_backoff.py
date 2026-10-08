"""Adaptive retraining schedule that backs off once the model has converged

Trains on the normal schedule (every ``retrain_every`` new results) while *active*. Once validation
loss stays below ``loss_threshold`` for ``consecutive_to_freeze`` rounds in a row, the thinker freezes:
training stops and the frozen model keeps serving inference -> simulation -> inference cycles. While
frozen, a "probe" retraining runs every ``probe_period`` new results (starting at ``probe_initial``).
If the probe's loss is still below threshold, the probe period backs off (multiplied by
``probe_backoff``, capped at ``probe_max``) and the thinker stays frozen. If the probe's loss exceeds
the threshold, the thinker returns to active and trains on the normal schedule again.
"""
import json
from pathlib import Path

from colmena.thinker import event_responder

from .incremental import IncrementalThinker


class ProbeBackoffThinker(IncrementalThinker):
    """An :class:`~examol.steer.incremental.IncrementalThinker` that stops retraining once the model
    has converged, periodically probing to check whether it should resume

    Args:
        loss_threshold: Validation loss (MSE) below which the model is considered converged
        consecutive_to_freeze: Number of consecutive converged training rounds before freezing
        probe_initial: Number of new results between probes right after freezing (``I0``)
        probe_backoff: Multiplier applied to the probe period after each probe that stays converged
        probe_max: Upper bound on the probe period
    """

    def __init__(self, *args, loss_threshold: float = 0.15, consecutive_to_freeze: int = 2,
                 probe_initial: int = 32, probe_backoff: float = 1.5, probe_max: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_threshold = loss_threshold
        self.consecutive_to_freeze = consecutive_to_freeze
        self.probe_initial = probe_initial
        self.probe_backoff = probe_backoff
        self.probe_max = probe_max

        self._base_retrain_every = self.retrain_every  # The normal (active-state) cadence
        self._state = 'active'  # 'active' or 'frozen'
        self._consecutive_below = 0
        self._probe_period = probe_initial

    @event_responder(event_name='start_training')
    def retrain(self):
        """Run a normal or probe retraining round, then update the active/frozen state"""
        self.retrain_every = self._base_retrain_every if self._state == 'active' else self._probe_period

        round_before = self._round
        IncrementalThinker.retrain.__wrapped__(self)
        if self._round == round_before:
            return  # The parent's guards decided there was nothing to do yet

        loss = self.last_val_loss
        state_before = self._state
        converged = loss is not None and loss < self.loss_threshold

        if state_before == 'active':
            self._consecutive_below = self._consecutive_below + 1 if converged else 0
            if self._consecutive_below >= self.consecutive_to_freeze:
                self._state = 'frozen'
                self._probe_period = self.probe_initial
        else:  # state_before == 'frozen': this round was a probe
            if converged:
                self._probe_period = min(self._probe_period * self.probe_backoff, self.probe_max)
            else:
                self._state = 'active'
                self._consecutive_below = 0
                self._probe_period = self.probe_initial

        self.logger.info(f'Probe-backoff: round={self._round} state {state_before}->{self._state} val_loss={loss} '
                         f'threshold={self.loss_threshold} consecutive_below={self._consecutive_below} next_period={self._probe_period:.1f}')
        with (Path(self.run_dir) / 'probe-backoff.json').open('a') as fp:
            print(json.dumps({
                'round': self._round, 'state_before': state_before, 'state_after': self._state,
                'val_loss': loss, 'threshold': self.loss_threshold, 'converged': converged,
                'consecutive_below': self._consecutive_below, 'retrain_every_used': self.retrain_every,
                'next_probe_period': self._probe_period,
            }), file=fp)
