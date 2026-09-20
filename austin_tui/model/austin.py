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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from austin.events import AustinMetadata
from austin.events import AustinSample
from austin.stats import AustinStats
from austin.stats import AustinStatsType

from austin_tui import AustinProfileMode


# Map TUI profile modes to the underlying Austin stats types.
_PROFILE_STATS_TYPES = {
    AustinProfileMode.TIME: AustinStatsType.WALL,
    AustinProfileMode.MEMORY: AustinStatsType.MEMORY_ALLOC,
}

# Map Austin "mode" metadata values to stats types.
_METADATA_STATS_TYPES = {
    "wall": AustinStatsType.WALL,
    "cpu": AustinStatsType.CPU,
    "memory": AustinStatsType.MEMORY_ALLOC,
}


class OrderedSet:
    """Ordered set."""

    def __init__(self) -> None:
        self._items: List[Any] = []
        self._map: Dict[Any, int] = {}

    def __contains__(self, element: Any) -> bool:
        """Check if the set contains the element."""
        return element in self._map

    def __getitem__(self, i: Any) -> Any:
        """Get the i-th item or the index of the given hashable object."""
        return self._items[i] if isinstance(i, int) else self._map[i]

    def __len__(self) -> int:
        """The number of elements in the set."""
        return len(self._items)

    def __iter__(self) -> Any:
        """Iterate over the items in insertion order."""
        return iter(self._items)

    def add(self, element: Any) -> None:
        """Add an element to the set.

        If the element is already in the set, nothing happens.
        """
        if element not in self._map:
            self._map[element] = len(self._items)
            self._items.append(element)

    def __bool__(self) -> bool:
        """Convert to boolean."""
        return bool(self._items)

    def __str__(self) -> str:
        """Representation of the set."""
        return type(self).__name__ + str(self._items)

    def __repr__(self) -> str:
        """Representation of the set."""
        return type(self).__name__ + repr(self._items)


class AustinModel:
    """Austin model.

    A model instance is a *revision segment* of the accumulated Austin data.
    While sampling, samples are reduced into the live segment. When the view is
    frozen, the current segment is sealed (it is never mutated again) and a new
    empty segment continues accumulating live data. On resume the segments are
    merged back into a single complete revision.
    """

    def __init__(self) -> None:
        self.mode: Optional[AustinProfileMode] = None

        self._samples = 0
        self._invalids = 0
        self._last_stack: Dict[str, AustinSample] = {}
        self._stats = AustinStats(AustinStatsType.WALL)

        self._austin_version: Optional[str] = None
        self._python_version: Optional[str] = None

        self._threads = OrderedSet()
        self._current_thread = 0

        self.metadata: Optional[Dict[str, str]] = None
        self.threshold = 0.0
        self.command_line: Optional[List[str]] = None

    def set_command_line(self, command_line: List[str]) -> None:
        """Set the command line."""
        self.command_line = command_line

    def get_versions(self) -> Tuple[Optional[str], Optional[str]]:
        """Get Austin and Python versions."""
        return self._austin_version, self._python_version

    def set_versions(self, austin_version: str, python_version: str) -> None:
        """Set Austin and Python versions."""
        self._austin_version = austin_version
        self._python_version = python_version

    def set_metadata(self, metadata: Dict[str, str]) -> None:
        """Set the Austin metadata."""
        self.metadata = metadata

    def _configure_stats_type(self, stats_type: AustinStatsType) -> None:
        """(Re)configure the stats type as long as no sample was reduced."""
        if (
            stats_type is not self._stats.stats_type
            and self._samples == 0
            and not self._stats.processes
        ):
            self._stats = AustinStats(stats_type)

    def set_mode(self, mode: AustinProfileMode) -> None:
        """Set the profiling mode, configuring the stats type if still empty."""
        self.mode = mode
        self._configure_stats_type(_PROFILE_STATS_TYPES[mode])

    def apply_metadata(self, metadata: AustinMetadata) -> None:
        """Apply a metadata event.

        Metadata is always reduced before any subsequent sample, so dependent
        configuration (like the stats type coming from the ``mode`` metadata)
        is in effect by the time the following samples are reduced.
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata[metadata.name] = metadata.value

        if metadata.name == "mode":
            stats_type = _METADATA_STATS_TYPES.get(metadata.value)
            if stats_type is not None:
                self._configure_stats_type(stats_type)

    def apply_sample(self, sample: AustinSample) -> Optional[str]:
        """Reduce a single sample into this segment.

        Returns the thread key that was touched, or ``None`` if the sample did
        not contribute to the stats.
        """
        try:
            # Negative time metrics are discarded, but they still count as
            # received samples (see the ``finally`` clause below).
            if sample.metrics.time is not None and sample.metrics.time < 0:
                return None
            self._stats.update(sample)
            thread_key = f"{sample.pid}:{sample.iid}:{sample.thread}"
            self._last_stack[thread_key] = sample
            self._threads.add(thread_key)
            return thread_key
        finally:
            self._samples += 1

    # --- Revision segments -------------------------------------------------

    def spawn_segment(self) -> "AustinModel":
        """Create a new empty live segment sharing this segment's config."""
        segment = AustinModel()
        segment.mode = self.mode
        # The metadata dictionary is shared: metadata can only grow and is
        # expected before samples, so in-place updates stay consistent.
        segment.metadata = self.metadata
        segment.command_line = self.command_line
        segment._austin_version = self._austin_version
        segment._python_version = self._python_version
        segment.threshold = self.threshold
        segment._current_thread = self._current_thread
        if self.mode is not None:
            segment._configure_stats_type(_PROFILE_STATS_TYPES[self.mode])
        return segment

    def absorb(self, segment: "AustinModel") -> None:
        """Merge a live ``segment`` accumulated while frozen into this one.

        This produces a single complete revision. The view state of the sealed
        revision (current thread and threshold) is preserved, so any thread
        navigation or threshold adjustment performed while paused survives the
        resume.
        """
        for pid, process in segment._stats.processes.items():
            dest_process = self._stats.processes.get(pid)
            if dest_process is None:
                self._stats.processes[pid] = process
                continue
            for thread_info, thread_stats in process.threads.items():
                if thread_info in dest_process.threads:
                    dest_process.threads[thread_info] << thread_stats
                else:
                    dest_process.threads[thread_info] = thread_stats

        for thread_key in segment._threads:
            self._threads.add(thread_key)
        self._last_stack.update(segment._last_stack)

        self._samples += segment._samples
        self._invalids += segment._invalids

    def get_last_stack(self, thread_key: str) -> AustinSample:
        """Get the last seen stack for the given thread."""
        return self._last_stack[thread_key]

    @property
    def stats(self) -> AustinStats:
        """The current Austin statistics."""
        return self._stats

    @property
    def threads(self) -> OrderedSet:
        """The seen threads as ordered set."""
        return self._threads

    @property
    def samples_count(self) -> int:
        """Get the sample count."""
        return self._samples

    @property
    def error_rate(self) -> float:
        """Get the error rate."""
        return self._invalids / self._samples

    @property
    def current_thread(self) -> int:
        """Get the currently active thread."""
        return self._current_thread

    @current_thread.setter
    def current_thread(self, n: int) -> None:
        """Set the currently active thread."""
        assert 0 <= n <= len(self._threads)
        self._current_thread = n
