from __future__ import annotations

from faro.agents.base import IntraExperimentAgent
from faro.core.data_structures import Channel, RTMSequence


class AdaptiveFrequencyAgent(IntraExperimentAgent):
    """Replace remaining events with fast acquisition when activity spikes.

    Usage pattern:

    1. User sends a **scouting sequence** (slow rate, many frames).
    2. This agent monitors each frame via :meth:`on_frame_processed`.
    3. When the mean value of *feature_column* in the latest frame exceeds
       *threshold*, the remaining scouting frames are replaced with a fast
       acquisition sequence.

    Example::

        agent = AdaptiveFrequencyAgent(
            feature_column="cnr",
            threshold=1.5,
            fast_interval=15.0,
            n_fast_frames=60,
            channels=(Channel(config="FITC", exposure=100),),
            stage_positions=[(x, y, z)],
        )
        ctrl = Controller(mic, pipeline, agent=agent)

        scouting = RTMSequence(
            time_plan={"interval": 30, "loops": 100},
            channels=[{"config": "FITC", "exposure": 100}],
            stage_positions=[(x, y, z)],
        )
        ctrl.run_experiment(list(scouting))
        ctrl.finish_experiment()

    Args:
        feature_column: Column name in ``df_tracked`` to monitor.
        threshold: Value above which the agent triggers.
        fast_interval: Interval (seconds) for the fast acquisition phase.
        n_fast_frames: Number of frames in the fast phase.
        channels: Channel tuple for the fast sequence.
        stage_positions: Stage positions for the fast sequence.
    """

    def __init__(
        self,
        *,
        feature_column: str = "cnr",
        threshold: float = 1.5,
        fast_interval: float = 15.0,
        n_fast_frames: int = 10,
        channels: tuple[Channel, ...] = (),
        stage_positions: list | None = None,
    ):
        super().__init__()
        self.feature_column = feature_column
        self.threshold = threshold
        self.fast_interval = fast_interval
        self.n_fast_frames = n_fast_frames
        self.channels = channels
        self.stage_positions = stage_positions
        self._triggered = False

    def on_frame_processed(self, result: dict) -> None:
        if self._triggered:
            return

        df = result["df_tracked"]
        if df.empty or self.feature_column not in df.columns:
            return

        latest_t = df["timestep"].max()
        latest_rows = df[df["timestep"] == latest_t]
        mean_val = latest_rows[self.feature_column].mean()

        if mean_val > self.threshold:
            self._triggered = True
            self._switch_to_fast()

    def _switch_to_fast(self) -> None:
        """Replace remaining scouting events with fast acquisition."""
        fast_sequence = RTMSequence(
            time_plan={"interval": self.fast_interval, "loops": self.n_fast_frames},
            channels=[
                {"config": ch.config, "exposure": ch.exposure} for ch in self.channels
            ],
            stage_positions=self.stage_positions or [],
        )
        self.controller.replace_remaining_events(list(fast_sequence))
