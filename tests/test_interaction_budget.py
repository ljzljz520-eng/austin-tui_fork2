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

"""R3: keyboard latency, queue high-water mark and zero drops under load."""

import asyncio
from time import perf_counter

from austin_tui.controller import INTERACTIVE_BUDGET
from austin_tui.model import Model
from tests.conftest import layout
from tests.conftest import make_sample


#: Number of distinct single-frame roots per thread (>= 4096 visible rows).
N_NODES = 4200
#: Samples produced between two keyboard interactions.
BURST = 256
PAUSED_BURST = 128
ROUNDS = 25

#: Disjoint frame-index range per thread so switching threads re-renders.
_THREAD_OFFSETS = {"T0": 0, "T1": N_NODES}


def _root_sample(thread: str, index: int) -> object:
    return make_sample(
        thread,
        _THREAD_OFFSETS[thread] + index,
        time=500 + index % 97,
        pid=10,
    )


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(pct * len(ordered)))]


def test_interactive_budget_under_sustained_load(controller) -> None:
    """Thread switching and pause/resume stay within the latency budget."""
    model: Model = controller.model

    # Two threads, each with far more visible nodes than 4096.
    async def populate() -> None:
        for thread in ("T0", "T1"):
            for index in range(N_NODES):
                await controller.on_sample(_root_sample(thread, index))

    asyncio.run(populate())
    model.drain_all()

    table_pad, _ = layout(controller)
    asyncio.run(controller.on_full_mode_selected())

    rows = controller.thread_full_data.transform()
    assert len(rows) >= 4096, len(rows)

    def burst(n: int, round_index: int) -> None:
        for _ in range(n // 2):
            model.submit(_root_sample("T0", round_index % N_NODES))
            model.submit(_root_sample("T1", round_index % N_NODES))

    latencies: dict[str, list[float]] = {
        "switch": [],
        "pause": [],
        "resume": [],
    }
    peak_pending = 0
    model._high_water = 0
    rendered_rows_before = table_pad.adds

    async def scenario() -> None:
        nonlocal peak_pending

        async def timed(key: str, coro) -> None:
            start = perf_counter()
            await coro
            latencies[key].append(perf_counter() - start)

        for round_index in range(ROUNDS):
            burst(BURST, round_index)
            peak_pending = max(peak_pending, model.queue_size)

            await timed("switch", controller.on_next_thread())
            await timed("switch", controller.on_previous_thread())

            await timed("pause", controller.on_play_pause())
            assert model.frozen

            burst(PAUSED_BURST, round_index)
            peak_pending = max(peak_pending, model.queue_size)

            await timed("resume", controller.on_play_pause())
            # The reducer commits the resume once the queue is fully drained.
            controller._tick()
            controller._tick()
            assert not model.frozen

    asyncio.run(scenario())

    # The table was actually rendered row by row on the fake pad, so the
    # measured handlers paid the full transform + render cost, including on
    # thread switches (the two threads have disjoint frame trees).
    assert table_pad.adds > rendered_rows_before

    p95_switch = _percentile(latencies["switch"], 0.95)
    p95_pause = _percentile(latencies["pause"], 0.95)
    p95_resume = _percentile(latencies["resume"], 0.95)

    assert p95_switch < INTERACTIVE_BUDGET, p95_switch
    assert p95_pause < INTERACTIVE_BUDGET, p95_pause
    assert p95_resume < INTERACTIVE_BUDGET, p95_resume

    # Queue high-water mark is controlled by the burst size: samples are
    # drained in bounded ticks and never accumulate without bound.
    assert model.queue_high_water <= BURST + PAUSED_BURST
    assert peak_pending <= BURST + PAUSED_BURST

    # No sample is silently dropped; everything submitted gets reduced.
    assert model.dropped_count == 0
    assert model.queue_size == 0
    assert model.reduced_count == model.submitted_count
