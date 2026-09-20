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

"""R1: batched reduction must be equivalent to per-event accumulation."""

import asyncio
from random import Random
from time import monotonic

from austin.events import AustinSample
from austin.stats import AustinStats
from austin.stats import AustinStatsType
from austin.stats import ThreadStats

from austin_tui.controller import AustinTUIController
from austin_tui.model import Model
from tests.conftest import inject
from tests.conftest import make_frame
from tests.conftest import make_metadata
from tests.conftest import make_sample


def _reference_update(reference: AustinStats, sample: AustinSample) -> None:
    """Mirror AustinModel.apply_sample on a per-event reference stats."""
    if sample.metrics.time is not None and sample.metrics.time < 0:
        # Negative wall/cpu time samples are discarded before reaching stats.
        return
    reference.update(sample)


def _iter_nodes(node: ThreadStats):
    yield node
    for child in node.children.values():
        yield from _iter_nodes(child)


def _assert_stats_equal(reduced: AustinStats, reference: AustinStats) -> None:
    """Compare two stats trees, cell by cell."""
    assert reduced.stats_type is reference.stats_type
    assert set(reduced.processes) == set(reference.processes)

    for pid, process in reduced.processes.items():
        reference_process = reference.processes[pid]
        assert set(process.threads) == set(reference_process.threads)

        for thread_info, thread in process.threads.items():
            reference_thread = reference_process.threads[thread_info]

            # Explicit OWN/TOTAL comparison at the thread root.
            assert thread.own == reference_thread.own, thread_info
            assert thread.total == reference_thread.total, thread_info

            # Explicit recursive call-tree comparison, including the frame
            # labels (which also pins insertion/merge order).
            got_nodes = list(_iter_nodes(thread))
            expected_nodes = list(_iter_nodes(reference_thread))
            assert len(got_nodes) == len(expected_nodes), thread_info
            for got, expected in zip(got_nodes, expected_nodes, strict=True):
                assert got.label == expected.label
                assert got.own == expected.own
                assert got.total == expected.total

    # Belt and braces: the generated dataclasses must compare equal as well.
    assert reduced == reference


def test_high_cardinality_stream_matches_reference(controller) -> None:
    """A high-cardinality stream reduces to the per-event reference."""
    rng = Random(20240517)
    events = [make_metadata("mode", "wall")]
    reference = AustinStats(AustinStatsType.WALL)

    n_samples = 0
    # 2 processes x 6 threads, random deep stacks from a shared frame pool
    # give far more than 4096 distinct call stacks while keeping merges
    # non-trivial.
    for _ in range(4800):
        pid = 10 + rng.randrange(2)
        thread = f"T{rng.randrange(6)}"
        depth = 1 + rng.randrange(8)
        frames = [make_frame(rng.randrange(256)) for _ in range(depth)]
        sample = make_sample(
            thread, frames, time=1 + rng.randrange(2000), pid=pid
        )
        events.append(sample)
        _reference_update(reference, sample)
        n_samples += 1

        # Sprinkle zero-time and negative-time samples: they must be counted
        # as received samples but must never contribute to the stats tree.
        if n_samples % 437 == 0:
            invalid = make_sample(thread, frames, time=-100, pid=pid)
            events.append(invalid)
            n_samples += 1
        if n_samples % 619 == 0:
            zero = make_sample(thread, frames, time=0, pid=pid)
            events.append(zero)
            _reference_update(reference, zero)
            n_samples += 1

    # Metadata interleaved with samples must be applied in queue order
    # without disturbing the accumulated stats.
    events.insert(2500, make_metadata("python", "3.12.0"))
    events.insert(4000, make_metadata("duration", "100000000"))

    asyncio.run(inject(controller, events))
    model: Model = AustinTUIController.model

    # Nothing is dropped on the way in.
    assert model.dropped_count == 0
    assert model.submitted_count == len(events)
    assert model.queue_size == len(events)

    # Drain through the bounded, time-boxed reducer rather than drain_all,
    # exercising the same path as the UI update loop.
    batches = 0
    while model.queue_size:
        result = model.reduce(max_events=37, deadline=monotonic() + 0.001)
        assert result.events > 0
        batches += 1
    assert batches > 1, "reduction should have spanned multiple batches"

    austin = model.austin
    assert model.queue_size == 0
    assert model.reduced_count == model.submitted_count
    assert model.dropped_count == 0
    assert austin.samples_count == n_samples
    assert model.revision > 1

    _assert_stats_equal(austin.stats, reference)

    # Metadata side effects were applied in FIFO order.
    assert austin.metadata is not None
    assert austin.metadata["mode"] == "wall"
    assert austin.metadata["python"] == "3.12.0"
    assert "Wall Time Profile" in str(controller.view.profile_mode.text)


def test_mode_metadata_precedes_dependent_samples(controller) -> None:
    """The stats type configured by mode metadata is set before samples."""
    events = [
        make_metadata("mode", "memory"),
        make_sample("T0", 1, time=None, pid=10, memory=4096),
        make_sample("T0", 2, time=None, pid=10, memory=8192),
        # Deallocation events must not contribute to the ALLOC profile.
        make_sample("T0", 3, time=None, pid=10, memory=-2048),
        make_sample("T0", 4, time=None, pid=10, memory=0),
        make_sample("T1", 5, time=None, pid=10, memory=16384),
    ]
    reference = AustinStats(AustinStatsType.MEMORY_ALLOC)
    for sample in events[1:]:
        reference.update(sample)

    asyncio.run(inject(controller, events))
    model: Model = AustinTUIController.model

    # Reduce just the metadata event first.
    first = model.reduce(max_events=1)
    assert first.events == 1
    assert model.queue_size == len(events) - 1

    # The metadata is effective *before* any of the dependent samples is
    # reduced.
    austin = model.austin
    assert austin.metadata is not None
    assert austin.metadata["mode"] == "memory"
    assert austin.stats.stats_type is AustinStatsType.MEMORY_ALLOC
    assert "Memory Profile" in str(controller.view.profile_mode.text)

    # Now drain the samples.
    model.drain_all()
    assert model.queue_size == 0
    assert austin.samples_count == 5
    _assert_stats_equal(austin.stats, reference)
