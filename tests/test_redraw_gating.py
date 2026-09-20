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

"""R2: redraws only happen for new revisions relevant to the current view."""

import asyncio

import pytest

from austin_tui.view.austin import AustinViewMode
from austin_tui.widgets import Point
from austin_tui.widgets import Rect
from austin_tui.widgets.graph import FlameGraph
from austin_tui.widgets.scroll import ScrollView
from austin_tui.widgets.table import Table
from tests.conftest import DrawSpy
from tests.conftest import FakePad
from tests.conftest import SetDataSpy
from tests.conftest import make_sample


# --- Widget level ------------------------------------------------------------


def _scroll_with(widget) -> tuple[ScrollView, FakePad]:
    scroll = ScrollView("scroll")
    scroll.add_child(widget)
    pad = FakePad()
    scroll._win = pad  # type: ignore[assignment]
    scroll.resize(Rect(0, Point(100 + 40j)))
    return scroll, pad


def test_table_redraws_only_when_data_changes() -> None:
    table = Table("table", 1)
    _, pad = _scroll_with(table)
    # The initial layout draws the (empty) table once.
    baseline = pad.clears
    assert baseline == 1

    table.set_data([["a"]])
    assert pad.clears == baseline + 1  # first population draws once

    # Repeated refresh cycles with unchanged data must not clear/redraw.
    table.draw()
    table.draw()
    assert pad.clears == baseline + 1

    # Equal data does not trigger a resize/redraw either.
    assert table.set_data([["a"]]) is False
    table.draw()
    assert pad.clears == baseline + 1

    # New data marks the table dirty; it is redrawn on the next draw cycle.
    assert table.set_data([["b"]]) is True
    table.draw()
    assert pad.clears == baseline + 2
    table.draw()
    assert pad.clears == baseline + 2

    # A forced draw (e.g. after a resize) is honoured.
    table.draw(force=True)
    assert pad.clears == baseline + 3


def test_flamegraph_redraws_only_when_data_changes() -> None:
    graph = FlameGraph("graph")
    _, pad = _scroll_with(graph)

    data: dict = {"root": (100.0, {})}
    graph.set_data(data)
    graph.draw()
    assert pad.clears == 1

    graph.draw()
    graph.draw()
    assert pad.clears == 1

    assert graph.set_data(dict(data)) is False
    graph.draw()
    assert pad.clears == 1

    graph.set_data({"root": (50.0, {}), "other": (50.0, {})})
    graph.draw()
    assert pad.clears == 2
    graph.draw()
    assert pad.clears == 2


# --- Controller level --------------------------------------------------------


async def _feed(controller, *samples) -> None:
    for sample in samples:
        await controller.on_sample(sample)


def _spy_transform(adapter) -> list:
    calls = []
    original = adapter.transform

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    adapter.transform = spy
    return calls


@pytest.fixture
def gating(controller) -> dict:
    """Controller with draw, set_data and transform spies attached."""
    draw = DrawSpy(controller)
    table_data = SetDataSpy(controller.view.table)
    graph_data = SetDataSpy(controller.view.flamegraph)
    transforms = {
        AustinViewMode.LIVE: _spy_transform(controller.thread_data),
        AustinViewMode.FULL: _spy_transform(controller.thread_full_data),
        AustinViewMode.TOP: _spy_transform(controller.thread_top_data),
        AustinViewMode.GRAPH: _spy_transform(controller.flamegraph),
    }
    return {
        "controller": controller,
        "draw": draw,
        "table_data": table_data,
        "graph_data": graph_data,
        "transforms": transforms,
    }


def _tick_expect(gating, changed, mode=AustinViewMode.LIVE):
    """Run one tick the way ``update_loop`` does.

    The data widget is drawn only when the tick reports a change; quiet
    ticks must not even reach the adapter transform or ``set_data``.
    """
    controller = gating["controller"]
    before_transforms = len(gating["transforms"][mode])
    before_table_data = gating["table_data"].calls
    before_graph_data = gating["graph_data"].calls
    before_table_draws = gating["draw"].table
    before_graph_draws = gating["draw"].flamegraph

    result = controller._tick()
    if result:
        # This is the only place update_loop issues a redraw.
        if mode is AustinViewMode.GRAPH:
            controller.view.flamegraph.draw()
        else:
            controller.view.table.draw()

    assert result is changed
    assert len(gating["transforms"][mode]) - before_transforms == (
        1 if changed else 0
    )
    if mode is AustinViewMode.GRAPH:
        assert gating["graph_data"].calls - before_graph_data == (
            1 if changed else 0
        )
        assert gating["draw"].flamegraph - before_graph_draws <= (
            1 if changed else 0
        )
    else:
        assert gating["table_data"].calls - before_table_data == (
            1 if changed else 0
        )
        # On quiet ticks the draw method is never even invoked.
        if not changed:
            assert gating["draw"].table == before_table_draws


def test_tick_revision_gating_live(gating) -> None:
    controller = gating["controller"]

    # No data yet: nothing to transform or draw.
    assert controller._tick() is False
    assert gating["table_data"].calls == 0
    assert gating["draw"].table == 0

    # First sample for the current (first) thread rebuilds once.
    asyncio.run(_feed(controller, make_sample("T0", [1, 2])))
    _tick_expect(gating, True)

    # Consecutive refresh cycles with the same revision are no-ops.
    _tick_expect(gating, False)
    _tick_expect(gating, False)
    _tick_expect(gating, False)
    revision = controller.model.revision
    assert controller._view_revision == revision

    # Revisions that only touch other threads do not rebuild the view.
    asyncio.run(
        _feed(
            controller,
            make_sample("T1", [3, 4]),
            make_sample("T2", [5, 6]),
        )
    )
    _tick_expect(gating, False)
    assert controller._view_revision == revision

    # A new revision touching the current thread rebuilds the view.
    asyncio.run(_feed(controller, make_sample("T0", [1, 7])))
    _tick_expect(gating, True)
    assert controller._view_revision == controller.model.revision

    # And quiet cycles stay quiet afterwards.
    _tick_expect(gating, False)


def test_tick_revision_gating_full_and_top(gating) -> None:
    controller = gating["controller"]

    asyncio.run(_feed(controller, make_sample("T0", [1, 2])))
    asyncio.run(
        _feed(
            controller,
            make_sample("T1", [3, 4]),
            make_sample("T2", [5, 6]),
        )
    )
    controller.model.drain_all()

    for mode, selector in (
        (AustinViewMode.FULL, controller.on_full_mode_selected),
        (AustinViewMode.TOP, controller.on_top_mode_selected),
    ):
        asyncio.run(selector())
        assert len(gating["transforms"][mode]) == 1

        # Same revision, refresh cycles: no further transforms.
        _tick_expect(gating, False, mode=mode)
        _tick_expect(gating, False, mode=mode)

        # Samples for other threads only do not rebuild the current view.
        asyncio.run(
            _feed(
                controller,
                make_sample("T1", [3, 8]),
                make_sample("T2", [9]),
            )
        )
        _tick_expect(gating, False, mode=mode)

        # A sample on the current thread rebuilds it.
        asyncio.run(_feed(controller, make_sample("T0", [1, 10])))
        _tick_expect(gating, True, mode=mode)
        _tick_expect(gating, False, mode=mode)


def test_tick_revision_gating_graph(gating) -> None:
    controller = gating["controller"]
    mode = AustinViewMode.GRAPH

    asyncio.run(_feed(controller, make_sample("T0", [1, 2])))
    asyncio.run(_feed(controller, make_sample("T1", [3, 4])))
    controller.model.drain_all()

    asyncio.run(controller.on_graph_selected())
    assert len(gating["transforms"][mode]) == 1
    assert gating["graph_data"].calls == 1

    _tick_expect(gating, False, mode=mode)

    asyncio.run(_feed(controller, make_sample("T1", [3, 11])))
    _tick_expect(gating, False, mode=mode)

    asyncio.run(_feed(controller, make_sample("T0", [1, 12])))
    _tick_expect(gating, True, mode=mode)
    _tick_expect(gating, False, mode=mode)


def test_thread_switch_rebuilds_only_selected_thread(gating) -> None:
    controller = gating["controller"]

    asyncio.run(
        _feed(
            controller,
            make_sample("T0", [1, 2]),
            make_sample("T1", [3, 4]),
            make_sample("T2", [5, 6]),
        )
    )
    controller.model.drain_all()
    controller._rebuild_view()
    live_transforms = gating["transforms"][AustinViewMode.LIVE]
    assert len(live_transforms) == 1
    assert controller.model.active_austin.current_thread == 0
    draws_after_first_build = gating["draw"].table

    # Switching threads rebuilds the newly selected thread's view and draws
    # it exactly once (plus at most a resize-forced draw).
    asyncio.run(controller.on_next_thread())
    assert controller.model.active_austin.current_thread == 1
    assert len(live_transforms) == 2
    assert 1 <= gating["draw"].table - draws_after_first_build <= 2

    # Further quiet ticks on the new thread do not rebuild.
    _tick_expect(gating, False)
    assert len(live_transforms) == 2

    # Navigating past the last thread is a no-op.
    asyncio.run(controller.on_next_thread())
    asyncio.run(controller.on_next_thread())
    assert controller.model.active_austin.current_thread == 2
    assert len(live_transforms) == 3

    asyncio.run(controller.on_previous_thread())
    assert controller.model.active_austin.current_thread == 1
    assert len(live_transforms) == 4

    asyncio.run(controller.on_previous_thread())
    asyncio.run(controller.on_previous_thread())
    assert controller.model.active_austin.current_thread == 0
    assert len(live_transforms) == 5
