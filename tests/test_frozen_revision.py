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
# This program is free software: you can redistribute it under the terms
# of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""R4: the frozen revision is pinned and resume switches atomically."""

import asyncio
from copy import deepcopy
from time import perf_counter

from austin.stats import AustinStats
from austin.stats import AustinStatsType

from austin_tui.model import Model
from austin_tui.model.system import FrozenSystemModel
from tests.conftest import layout
from tests.conftest import make_sample


N_NODES = 4200

#: Disjoint frame-index range per thread so that per-thread views differ.
_THREAD_OFFSETS = {"T0": 0, "T1": N_NODES, "T9": 2 * N_NODES}


def _root_sample(thread: str, index: int, time: int = 1000):
    return make_sample(
        thread, _THREAD_OFFSETS[thread] + index, time=time, pid=10
    )


async def _populate(controller, n_nodes: int) -> None:
    for thread in ("T0", "T1"):
        for index in range(n_nodes):
            await controller.on_sample(
                _root_sample(thread, index, time=500 + index % 97)
            )


def _all_sample_streams():
    """Reproduce the complete sample stream for the reference stats."""
    samples = []
    for thread in ("T0", "T1"):
        for index in range(N_NODES):
            samples.append(_root_sample(thread, index, time=500 + index % 97))
    # Samples produced while paused.
    for index in range(64):
        samples.append(_root_sample("T0", index, time=1000))
    # A brand new thread that only exists in the live segment.
    for index in range(8):
        samples.append(_root_sample("T9", index, time=1000))
    return samples


def test_pause_is_o1_and_pins_frozen_revision(controller) -> None:
    model: Model = controller.model
    asyncio.run(_populate(controller, N_NODES))
    model.drain_all()

    layout(controller)
    asyncio.run(controller.on_full_mode_selected())

    live_stats = model.austin.stats
    samples_before = model.austin.samples_count
    rows_before = controller.thread_full_data.transform()
    assert len(rows_before) >= 4096

    # Pausing a multi-thousand-node model must not deep copy and must be
    # effectively instant.
    start = perf_counter()
    asyncio.run(controller.on_play_pause())
    pause_seconds = perf_counter() - start
    assert pause_seconds < 0.05, pause_seconds

    assert model.frozen
    sealed = model.frozen_austin
    assert sealed is not None
    # O(1) seal: the sealed stats are the very same object, never a copy.
    assert sealed.stats is live_stats
    assert model.austin.stats is not live_stats
    assert sealed.samples_count == samples_before

    # The system snapshot is frozen too, including the child process.
    frozen_system = model.frozen_system
    assert isinstance(frozen_system, FrozenSystemModel)
    assert model.active_system is frozen_system
    model.system.set_child_process(object())
    assert frozen_system.child_process is None
    assert model.active_system is frozen_system

    sealed_stats_snapshot = deepcopy(sealed.stats)

    # Live data keeps growing while the screen reads the sealed revision.
    revision_at_freeze = model.revision
    for index in range(64):
        model.submit(_root_sample("T0", index, time=1000))
    for index in range(8):
        model.submit(_root_sample("T9", index, time=1000))

    assert controller._tick() is False  # frozen: nothing to (re)draw
    assert model.frozen
    assert model.revision > revision_at_freeze
    assert model.austin.samples_count == 72

    # The sealed revision is untouched: identity, counts and rows.
    assert model.active_austin is sealed
    assert sealed.samples_count == samples_before
    assert sealed.stats == sealed_stats_snapshot
    assert controller.thread_full_data.transform() == rows_before

    # Threshold recomputation reads the sealed revision: raising it filters
    # the sealed rows, while the live segment keeps its own threshold.
    sealed.threshold = 0.5
    assert controller.thread_full_data.transform() == []
    assert model.austin.threshold == 0.0
    sealed.threshold = 0.0
    assert controller.thread_full_data.transform() == rows_before

    # Thread navigation reads the sealed revision: T9 (live only) is not
    # reachable, and navigation cannot wrap past the last sealed thread.
    assert sealed.current_thread == 0
    assert asyncio.run(controller.on_next_thread()) is True
    assert sealed.current_thread == 1
    assert asyncio.run(controller.on_next_thread()) is False
    assert sealed.current_thread == 1
    assert asyncio.run(controller.on_previous_thread()) is True
    assert sealed.current_thread == 0

    # Leave a non-trivial threshold and the navigation in place to verify
    # they survive the resume merge below.
    sealed.threshold = 0.25
    asyncio.run(controller.on_next_thread())
    assert sealed.current_thread == 1


def test_resume_atomically_publishes_complete_revision(controller) -> None:
    model: Model = controller.model

    reference = AustinStats(AustinStatsType.WALL)
    for sample in _all_sample_streams():
        reference.update(sample)

    asyncio.run(_populate(controller, N_NODES))
    model.drain_all()
    layout(controller)
    asyncio.run(controller.on_full_mode_selected())

    asyncio.run(controller.on_play_pause())
    sealed = model.frozen_austin
    assert sealed is not None
    samples_before_pause = sealed.samples_count

    # Live accumulation while paused.
    for index in range(64):
        model.submit(_root_sample("T0", index, time=1000))
    for index in range(8):
        model.submit(_root_sample("T9", index, time=1000))
    model.reduce()
    live_samples = model.austin.samples_count
    assert live_samples == 72

    # Paused view state.
    sealed.threshold = 0.25
    sealed.current_thread = 1

    # Request resume; the commit is deferred until the queue is drained so
    # that the published revision is complete.
    model.request_resume()
    result = model.reduce()
    assert result.resumed
    assert not model.frozen

    # Atomic switch: a single consistent revision is now visible.
    assert model.frozen_austin is None
    assert model.frozen_system is None
    assert model.active_austin is sealed
    assert model.queue_size == 0
    assert sealed.samples_count == samples_before_pause + live_samples
    assert sealed.stats == reference
    assert sealed.samples_count == len(_all_sample_streams())

    # No mixed counts: per-thread totals are exactly the sum of the two
    # segments.
    process = sealed.stats.processes[10]
    for thread_info, thread in process.threads.items():
        reference_thread = reference.processes[10].threads[thread_info]
        assert thread.total == reference_thread.total
        assert thread.own == reference_thread.own

    # Paused view state is preserved across the resume.
    assert sealed.threshold == 0.25
    assert sealed.current_thread == 1

    # The live-only thread is now navigable.
    sealed.threshold = 0.0
    names = [info.thread for info in process.threads]
    assert "T9" in names
    reached = sealed.current_thread
    while asyncio.run(controller.on_next_thread()):
        reached = sealed.current_thread
    assert sealed.threads[reached] == "10:0:T9"
