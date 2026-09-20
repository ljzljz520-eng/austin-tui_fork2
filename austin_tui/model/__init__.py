# This file is part of "austin-tui" which is released under GPL.
#
# See file LICENCE or go to http://www.gnu.org/licenses/ for full license
# details.
#
# austin-tui is top-like TUI for Austin.
#
# Copyright (c) 2018-2020 Gabriele N. Tornetta <phoenix1987@gmail.com>.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Callable
from typing import Deque
from typing import FrozenSet
from typing import Optional
from typing import Union

from austin.events import AustinEvent
from austin.events import AustinMetadata
from austin.events import AustinSample

from austin_tui.model.austin import AustinModel
from austin_tui.model.system import FrozenSystemModel
from austin_tui.model.system import SystemModel


#: Maximum number of events reduced in one tick by default.
DEFAULT_REDUCER_BATCH = 1 << 12
#: Maximum wall-clock time (seconds) spent reducing in one tick by default.
DEFAULT_REDUCER_BUDGET = 0.02


@dataclass(frozen=True)
class ReduceResult:
    """The outcome of a reducer drain."""

    events: int = 0
    samples: int = 0
    dirty: FrozenSet[str] = frozenset()
    remaining: int = 0
    froze: bool = False
    resumed: bool = False


class Model:
    """The application model.

    Austin events (samples and metadata) are submitted to an ordered queue and
    reduced in batches by :meth:`reduce`. Reduction always happens on the same
    event loop as the UI, but the amount of work per tick is bounded so that
    keyboard handling stays responsive. Events are never dropped: the submitted
    and reduced counters, together with the queue high-water mark, are exposed
    for observability.

    Freezing seals the latest complete revision in O(1) time: the live segment
    is rotated instead of being deep-copied. Samples received while frozen
    keep accumulating in the new live segment, while the view keeps reading the
    sealed revision. Resuming atomically merges the live segment back into the
    sealed revision at a drain boundary.
    """

    __slots__ = (
        "austin",
        "system",
        "frozen_austin",
        "frozen_system",
        "frozen",
        "_queue",
        "_submitted",
        "_reduced",
        "_high_water",
        "_revision",
        "_freeze_requested",
        "_resume_requested",
        "metadata_callback",
    )

    _instance: Optional["Model"] = None

    @classmethod
    def get(cls) -> "Model":
        """Get the single model instance."""
        if cls._instance is not None:
            return cls._instance

        model = cls._instance = cls()
        return model

    def __init__(self) -> None:
        self.austin = AustinModel()
        self.system = SystemModel()
        self.frozen_austin: Optional[AustinModel] = None
        self.frozen_system: Optional[FrozenSystemModel] = None
        self.frozen = False

        self._queue: Deque[AustinEvent] = deque()
        self._submitted = 0
        self._reduced = 0
        self._high_water = 0
        self._revision = 0

        self._freeze_requested = False
        self._resume_requested = False

        self.metadata_callback: Optional[Callable[[AustinMetadata], None]] = (
            None
        )

    # --- Event queue -------------------------------------------------------

    def submit(self, event: AustinEvent) -> None:
        """Submit an Austin event to the reduction queue."""
        self._queue.append(event)
        self._submitted += 1
        queued = len(self._queue)
        if queued > self._high_water:
            self._high_water = queued

    @property
    def queue_size(self) -> int:
        """The number of events waiting to be reduced."""
        return len(self._queue)

    @property
    def submitted_count(self) -> int:
        """The total number of events submitted to the queue."""
        return self._submitted

    @property
    def reduced_count(self) -> int:
        """The total number of events reduced so far."""
        return self._reduced

    @property
    def dropped_count(self) -> int:
        """The number of events that were silently dropped (always zero)."""
        return 0

    @property
    def queue_high_water(self) -> int:
        """The largest size the reduction queue ever reached."""
        return self._high_water

    @property
    def revision(self) -> int:
        """The current data revision.

        The revision advances once per drain that reduces at least one sample,
        i.e. it identifies complete revisions rather than individual samples.
        """
        return self._revision

    # --- Active (visible) revision ----------------------------------------

    @property
    def active_austin(self) -> AustinModel:
        """The Austin model revision the view must read."""
        if self.frozen:
            assert self.frozen_austin is not None
            return self.frozen_austin
        return self.austin

    @property
    def active_system(
        self,
    ) -> Union[SystemModel, FrozenSystemModel]:
        """The system model revision the view must read."""
        if self.frozen:
            assert self.frozen_system is not None
            return self.frozen_system
        return self.system

    # --- Freeze / resume ---------------------------------------------------

    def request_freeze(self) -> None:
        """Request the model to be frozen at the next drain boundary."""
        if not self.frozen:
            self._freeze_requested = True

    def request_resume(self) -> None:
        """Request resuming at the next drain boundary.

        The resume is committed only once the queue is fully drained so that
        the revision published to the view is complete.
        """
        if self.frozen:
            self._resume_requested = True

    def _commit_freeze(self) -> None:
        """Seal the live revision and rotate to a fresh live segment."""
        sealed = self.austin
        self.frozen_austin = sealed
        self.frozen_system = self.system.freeze()
        self.austin = sealed.spawn_segment()
        self.frozen = True
        self._freeze_requested = False

    def _commit_resume(self) -> None:
        """Merge the live segment into the sealed revision and publish it."""
        assert self.frozen_austin is not None
        sealed = self.frozen_austin
        sealed.absorb(self.austin)
        self.austin = sealed
        self.frozen_austin = None
        self.frozen_system = None
        self.frozen = False
        self._resume_requested = False

    # --- Reducer -----------------------------------------------------------

    def reduce(
        self,
        max_events: Optional[int] = DEFAULT_REDUCER_BATCH,
        deadline: Optional[float] = None,
    ) -> ReduceResult:
        """Drain queued events in FIFO order and publish one revision.

        ``max_events`` bounds the number of events reduced during this call and
        ``deadline`` bounds the wall-clock time spent. Metadata events are
        applied before the samples that follow them in the queue. Pending
        freeze/resume requests are committed atomically at the drain boundary.
        """
        events = 0
        samples = 0
        dirty = set()

        queue = self._queue
        austin = self.austin
        metadata_callback = self.metadata_callback
        while queue:
            if max_events is not None and events >= max_events:
                break
            if deadline is not None and monotonic() >= deadline:
                break

            event = queue.popleft()
            events += 1
            if isinstance(event, AustinSample):
                thread_key = austin.apply_sample(event)
                samples += 1
                if thread_key is not None:
                    dirty.add(thread_key)
            elif isinstance(event, AustinMetadata):
                austin.apply_metadata(event)
                if metadata_callback is not None:
                    metadata_callback(event)

        self._reduced += events
        if samples:
            self._revision += 1

        froze = resumed = False
        if self._resume_requested and self.frozen and not queue:
            self._commit_resume()
            resumed = True
        elif self._freeze_requested and not self.frozen:
            self._commit_freeze()
            froze = True

        return ReduceResult(
            events=events,
            samples=samples,
            dirty=frozenset(dirty),
            remaining=len(queue),
            froze=froze,
            resumed=resumed,
        )

    def drain_all(self) -> ReduceResult:
        """Reduce every queued event, ignoring batch and time bounds."""
        return self.reduce(max_events=None, deadline=None)
