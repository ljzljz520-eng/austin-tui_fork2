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

"""Shared fixtures and helpers for the reducer/redraw/freeze test suites."""

import curses
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

import pytest
from austin.events import AustinFrame
from austin.events import AustinMetadata
from austin.events import AustinMetrics
from austin.events import AustinSample

from austin_tui.controller import AustinTUIController
from austin_tui.model import Model
from austin_tui.widgets import Rect
from tests.mcurses import MWindow


# ACS characters are only available after curses initialisation, which the
# headless test suite never performs. Provide harmless stand-ins so that the
# scroll-bar rendering code paths can still execute.
for _acs_name in ("ACS_VLINE", "ACS_CKBOARD"):
    if not hasattr(curses, _acs_name):
        setattr(curses, _acs_name, 0)


@pytest.fixture
def model() -> Iterable[Model]:
    """A fresh application model wired onto the controller class."""
    Model._instance = None
    fresh = Model()
    AustinTUIController.model = fresh
    yield fresh
    Model._instance = None


@pytest.fixture
def controller(model: Model) -> AustinTUIController:
    """A controller bound to a fresh model.

    The system duration is pinned so that time-scaling adapters always have a
    non-zero denominator.
    """
    austin_controller = AustinTUIController()
    model.system._duration = 100.0
    return austin_controller


def make_frame(index: int, module: str = "mod") -> AustinFrame:
    """Build a deterministic frame for the given (unique) index."""
    return AustinFrame(
        filename=f"{module}_{index % 32}.py",
        function=f"func_{index}",
        line=(index % 200) + 1,
        column=1,
    )


def make_sample(
    thread: str,
    frames: Union[int, Iterable[Union[int, AustinFrame]]],
    time: Optional[int] = 1000,
    pid: int = 4242,
    iid: int = 0,
    memory: Optional[int] = None,
) -> AustinSample:
    """Build a sample.

    ``frames`` may be a list of frames or a single integer, in which case a
    one-frame stack is generated.
    """
    if isinstance(frames, int):
        frame_stack = (make_frame(frames),)
    else:
        frame_stack = tuple(
            make_frame(frame) if isinstance(frame, int) else frame
            for frame in frames
        )
    return AustinSample(
        pid=pid,
        iid=iid,
        thread=thread,
        metrics=AustinMetrics(time=time, memory=memory),
        frames=frame_stack,
    )


def make_metadata(name: str, value: str) -> AustinMetadata:
    """Build a metadata event."""
    return AustinMetadata(name, value)


async def inject(
    austin_controller: AustinTUIController, events: Iterable[object]
) -> None:
    """Push events through the controller callbacks (the public entry point)."""
    for event in events:
        if isinstance(event, AustinSample):
            await austin_controller.on_sample(event)
        else:
            await austin_controller.on_metadata(event)  # type: ignore[arg-type]


class FakePad:
    """A minimal recording replacement for a curses pad."""

    def __init__(self) -> None:
        self.clears = 0
        self.adds = 0
        self.refreshes = 0

    def clear(self) -> None:
        self.clears += 1

    def addstr(self, *args: object, **kwargs: object) -> None:
        self.adds += 1

    def refresh(self, *args: object, **kwargs: object) -> None:
        self.refreshes += 1

    def vline(self, *args: object, **kwargs: object) -> None:
        pass

    def resize(self, *args: object, **kwargs: object) -> None:
        pass

    def nodelay(self, *args: object, **kwargs: object) -> None:
        pass


def layout(
    austin_controller: AustinTUIController, width: int = 80, height: int = 32
) -> tuple[FakePad, FakePad]:
    """Lay out the real view tree and attach fake pads to the scroll views."""
    root = austin_controller.view.root_widget
    assert root is not None
    root._win = MWindow(width, height)
    root.resize(Rect(0, root.get_size()))

    table_pad = FakePad()
    graph_pad = FakePad()
    austin_controller.view.stats_view._win = table_pad  # type: ignore[assignment]
    austin_controller.view.flame_view._win = graph_pad  # type: ignore[assignment]
    return table_pad, graph_pad


class DrawSpy:
    """Count draw invocations on data widgets."""

    def __init__(self, austin_controller: AustinTUIController) -> None:
        self.table = 0
        self.flamegraph = 0
        table_widget = austin_controller.view.table
        graph_widget = austin_controller.view.flamegraph
        table_draw = table_widget.draw
        graph_draw = graph_widget.draw

        def table_draw_spy(force: bool = False) -> bool:
            self.table += 1
            return bool(table_draw(force=force))

        def graph_draw_spy(force: bool = False) -> bool:
            self.flamegraph += 1
            return bool(graph_draw(force=force))

        table_widget.draw = table_draw_spy  # type: ignore[method-assign]
        graph_widget.draw = graph_draw_spy  # type: ignore[method-assign]


class SetDataSpy:
    """Count ``set_data`` invocations, i.e. actual view rebuilds."""

    def __init__(self, widget: Any) -> None:
        self.calls = 0
        self._original = widget.set_data

        def spy(data: Any) -> bool:
            self.calls += 1
            return bool(self._original(data))

        widget.set_data = spy
