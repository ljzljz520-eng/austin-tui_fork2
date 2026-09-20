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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import sys
from enum import Enum
from pathlib import Path
from textwrap import wrap
from time import monotonic
from time import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence

from austin.aio import AsyncAustin
from austin.cli import AustinArgumentParser
from austin.cli import AustinCommandLineError
from austin.errors import AustinError
from austin.events import AustinMetadata
from austin.events import AustinSample
from austin.format.mojo import MojoStreamReader
from austin.format.mojo import MojoStreamWriter
from psutil import Process

from austin_tui import AustinProfileMode
from austin_tui.adapters import Adapter
from austin_tui.adapters import CommandLineAdapter
from austin_tui.adapters import CountAdapter
from austin_tui.adapters import CpuAdapter
from austin_tui.adapters import CurrentThreadAdapter
from austin_tui.adapters import DurationAdapter
from austin_tui.adapters import FlameGraphAdapter
from austin_tui.adapters import MemoryAdapter
from austin_tui.adapters import ThreadDataAdapter
from austin_tui.adapters import ThreadFullDataAdapter
from austin_tui.adapters import ThreadNameAdapter
from austin_tui.adapters import ThreadTopDataAdapter
from austin_tui.model import Model
from austin_tui.view import ViewBuilder
from austin_tui.view.austin import AustinView
from austin_tui.view.austin import AustinViewMode
from austin_tui.widgets.markup import escape


class ThreadNav(Enum):
    """Thread navigation."""

    PREV = -1
    NEXT = 1


#: Interval between UI refresh ticks (seconds).
UPDATE_INTERVAL = 0.05
#: Maximum number of queued Austin events reduced in a single tick.
REDUCER_BATCH = 1 << 12
#: Maximum wall-clock time (seconds) spent reducing events in a single tick.
REDUCER_BUDGET = 0.02
#: Interactive latency budget for keyboard events: p95 must stay below it.
INTERACTIVE_BUDGET = 0.1


def _print(text: str) -> None:
    for line in wrap(text, 78):
        print(line, file=sys.stderr)


class AustinTUIArgumentParser(AustinArgumentParser):
    """Austin TUI implementation of the Austin argument parser."""

    def __init__(self) -> None:
        super().__init__(name="austin-tui", full=False)

        self.add_argument(
            "-o",
            "--open",
            help="Open a MOJO file",
            type=Path,
        )

    def parse_args(self) -> Any:
        """Parse command line arguments and report any errors."""
        try:
            return super().parse_args()
        except AustinCommandLineError as e:
            reason, *code = e.args
            # If --open was given, the PID/command requirement doesn't apply.
            args, _ = super().parse_known_args()
            if getattr(args, "open", None) is not None:
                return args
            if reason:
                _print(reason)
            exit(code[0] if code else -1)


class AustinTUIController:
    """Austin controller.

    This controller is in charge of Austin data managing and UI updates.
    """

    model = Model.get()  # type: ignore[assignment]

    cpu = CpuAdapter
    memory = MemoryAdapter
    duration = DurationAdapter
    samples = CountAdapter
    current_thread = CurrentThreadAdapter
    thread_name = ThreadNameAdapter
    thread_data = ThreadDataAdapter
    thread_full_data = ThreadFullDataAdapter
    thread_top_data = ThreadTopDataAdapter
    command_line = CommandLineAdapter
    flamegraph = FlameGraphAdapter

    def __init__(self) -> None:
        self._view_mode = AustinViewMode.LIVE
        self._scaler: Optional[Callable[..., Any]] = None
        self._formatter: Optional[Callable[..., Any]] = None
        self._update_task: Optional[asyncio.Task[None]] = None
        self._exception: Optional[Exception] = None
        self._file_mode = False

        # Revision gating: the current thread view is rebuilt only when the
        # data revision advanced with changes relevant to the current thread,
        # or when a rebuild is explicitly forced (thread/mode/threshold).
        self._pending_dirty: set[str] = set()
        self._view_revision = -1
        self._force_view = True

        view_builder = ViewBuilder.from_resource(
            "austin_tui.view", "tui.austinui"
        )

        self.austin: Optional[AsyncAustin] = None
        self.view: AustinView = view_builder.build()  # type: ignore[assignment]
        view = self.view
        self.view.callback = self.on_view_event

        view_builder.autoconnect(self)

        self.model.metadata_callback = self._on_metadata_reduced
        self.model.austin.set_mode(view.mode)

        # Auto-create adapters
        for name, adapter_class in (
            (n, v)
            for n, v in type(self).__dict__.items()
            if isinstance(v, type) and v.__mro__[-2] == Adapter
        ):
            setattr(self, name, adapter_class(self.model, self.view))

    def set_thread_data(self) -> bool:
        """Set the thread stack for the active revision.

        Returns whether the underlying widget data actually changed.
        """
        if not self.model.active_austin.threads:
            return False

        if self._view_mode is AustinViewMode.GRAPH:
            return bool(self.flamegraph())  # type: ignore[call-arg]
        elif self._view_mode is AustinViewMode.FULL:
            return bool(self.thread_full_data())  # type: ignore[call-arg]
        elif self._view_mode is AustinViewMode.TOP:
            return bool(self.thread_top_data())  # type: ignore[call-arg]
        else:
            return bool(self.thread_data())  # type: ignore[call-arg]

    def set_thread(self) -> bool:
        """Set the thread to display."""
        self.current_thread()  # type: ignore[call-arg]
        self.thread_name()

        # Populate the thread stack view
        return self.set_thread_data()

    def _rebuild_view(self) -> bool:
        """Rebuild the current thread view and mark the revision as rendered."""
        changed = self.set_thread()

        active = self.model.active_austin
        if active.threads:
            self._pending_dirty.discard(active.threads[active.current_thread])
        self._view_revision = self.model.revision
        self._force_view = False

        return changed

    def _render_view(self) -> None:
        """Rebuild the active view and draw the visible data widget."""
        self._rebuild_view()
        if self._view_mode is AustinViewMode.GRAPH:
            self.view.flamegraph.draw()
            self.view.flame_view.refresh()
        else:
            self.view.table.draw()
            self.view.stats_view.refresh()

    def _tick(self) -> bool:
        """Run one UI update cycle.

        Reduce a bounded batch of queued events, refresh the cheap header
        labels, and rebuild the (potentially expensive) thread view only when
        the current revision brought changes relevant to the current thread.
        """
        result = self.model.reduce(
            max_events=REDUCER_BATCH,
            deadline=monotonic() + REDUCER_BUDGET,
        )
        self._pending_dirty.update(result.dirty)

        if result.froze:
            self._force_view = True

        if self.model.frozen:
            # The visible revision is sealed: nothing live can dirty it. The
            # only transition that requires a rebuild is a resume commit.
            if not result.resumed:
                return False
            self._force_view = True

        if result.resumed:
            self.view.notification.set_text("Resumed")

        # System data
        self.duration()
        self.cpu()  # type: ignore[call-arg]
        self.memory()  # type: ignore[call-arg]

        # Samples count and thread indicators
        self.samples()
        self.current_thread()  # type: ignore[call-arg]
        self.thread_name()

        active = self.model.active_austin
        if not active.threads:
            return False

        if not self._force_view:
            current_key = active.threads[active.current_thread]
            if (
                self.model.revision == self._view_revision
                or current_key not in self._pending_dirty
            ):
                return False

        return self._rebuild_view()

    def _add_flamegraph_palette(self) -> None:
        colors = [196, 202, 214, 124, 160, 166, 208]
        palette = self.view.palette

        for i, color in enumerate(colors):
            palette.add_color(f"fg{i}", 15, color)
            palette.add_color(f"fgf{i}", color)

        self.view.flamegraph.set_palette(
            (
                [palette.get_color(f"fg{i}") for i in range(len(colors))],
                [palette.get_color(f"fgf{i}") for i in range(len(colors))],
            )
        )

    async def start(self, args: Sequence[str]) -> None:
        """Start event."""
        pargs = AustinTUIArgumentParser().parse_args()  # type: ignore[call-arg]

        if pargs.open is not None and pargs.open.exists():
            await self.open_file(pargs.open)
            return

        self.austin = AsyncAustin(
            self.on_sample, self.on_metadata, self.on_terminate
        )

        await self.austin.start(args)

        if pargs.pid is not None:
            child_process = Process(pargs.pid)
        else:
            austin_process = Process(self.austin._proc.pid)
            (child_process,) = austin_process.children()
        command = child_process.cmdline()

        mode = (
            AustinProfileMode.MEMORY if pargs.memory else AustinProfileMode.TIME
        )
        self.view.mode = mode
        self.model.austin.set_mode(mode)

        """Austin ready callback."""
        self.model.system.set_child_process(child_process)
        self.model.austin.set_command_line(command)

        self._add_flamegraph_palette()
        self.view.open()
        self._update_task = asyncio.create_task(self.update_loop())

        self._formatter, self._scaler = (
            (self.view.fmt_mem, self.view.scale_memory)
            if self.view.mode == AustinProfileMode.MEMORY
            else (self.view.fmt_time, self.view.scale_time)
        )
        self.model.system.start()

        self.command_line()

        self.view.set_pid(child_process.pid, pargs.children)

        try:
            await self.austin.wait()
        except Exception:
            self.shutdown()
            raise

        try:
            if self.view._input_task is not None:
                await self.view._input_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.shutdown()
            raise

        if self._exception is not None:
            raise self._exception

    async def open_file(self, path: Path) -> None:
        """Open a MOJO file and replay its events into the TUI."""
        print(f"📂 Opening MOJO file '{path}' ...", end="", flush=True)
        try:
            with path.open("rb") as f:
                mojo = MojoStreamReader(f)
                for event in mojo:
                    if isinstance(event, AustinSample):
                        await self.on_sample(event)
                    elif isinstance(event, AustinMetadata):
                        await self.on_metadata(event)
        except AustinError as e:
            self.shutdown()
            _print(f"❌ Failed to open MOJO file '{path}': {e}")
            exit(-1)

        self.model.austin.set_command_line(["<MOJO file>", str(path)])

        self._view_mode = AustinViewMode.FULL
        self._file_mode = True

        self._add_flamegraph_palette()
        self.view.open()
        self.view.on_mode_selected(AustinViewMode.FULL)
        self.view.live_mode_cmd.set_color("disabled")
        self.view.save_cmd.set_color("disabled")
        self._update_task = asyncio.create_task(self.update_loop())

        self._formatter, self._scaler = (
            (self.view.fmt_mem, self.view.scale_memory)
            if self.view.mode == AustinProfileMode.MEMORY
            else (self.view.fmt_time, self.view.scale_time)
        )

        self.command_line()
        self.model.drain_all()
        self._render_view()

        await self.stop()

        try:
            if self.view._input_task is not None:
                await self.view._input_task
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Called when Austin exits: cancel the update task and mark the view stopped.

        Does not close the view — the user can still review final stats and press Q.
        """
        self.model.system.stop()

        if self._update_task is not None:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._exception = exc
            self._update_task = None

        self.view.stop()

    async def update_loop(self) -> None:
        """The UI update loop."""
        try:
            while (
                not self.view._stopped
                and self.view.is_open
                and self.view.root_widget
            ):
                if self._tick():
                    if self._view_mode is AustinViewMode.GRAPH:
                        self.view.flamegraph.draw()
                    else:
                        self.view.table.draw()

                self.view.root_widget.refresh()

                try:
                    await asyncio.sleep(UPDATE_INTERVAL)
                except asyncio.CancelledError:
                    break
        except Exception as exc:
            self.view.on_exception(exc)

    def _change_thread(self, direction: ThreadNav) -> bool:
        """Change thread on the active (live or frozen) revision."""
        austin = self.model.active_austin
        prev_index = austin.current_thread

        austin.current_thread = max(
            0,
            min(
                austin.current_thread + direction.value,
                len(austin.threads) - 1,
            ),
        )

        if prev_index != austin.current_thread:
            self._force_view = True
            return self._rebuild_view()

        return False

    async def on_next_thread(self) -> bool:
        """Handle next thread event."""
        if self._change_thread(ThreadNav.NEXT):
            if self._view_mode is AustinViewMode.GRAPH:
                self.view.flamegraph.draw()
                self.view.flame_view.refresh()
            else:
                self.view.table.draw()
                self.view.stats_view.refresh()
            return True
        return False

    async def on_previous_thread(self) -> bool:
        """Handle previous thread event."""
        if self._change_thread(ThreadNav.PREV):
            if self._view_mode is AustinViewMode.GRAPH:
                self.view.flamegraph.draw()
                self.view.flame_view.refresh()
            else:
                self.view.table.draw()
                self.view.stats_view.refresh()
            return True
        return False

    async def on_live_mode_selected(self, _: Any = None) -> bool:
        """Select live mode."""
        if self._file_mode or self._view_mode is AustinViewMode.LIVE:
            return False

        self._view_mode = AustinViewMode.LIVE
        self.view.dataview_selector.select(0)
        self._rebuild_view()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_top_mode_selected(self, _: Any = None) -> bool:
        """Select top mode."""
        if self._view_mode is AustinViewMode.TOP:
            return False

        self._view_mode = AustinViewMode.TOP
        self.view.dataview_selector.select(0)
        self._rebuild_view()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_full_mode_selected(self, _: Any = None) -> bool:
        """Toggle full mode."""
        if self._view_mode is AustinViewMode.FULL:
            return False

        self._view_mode = AustinViewMode.FULL
        self.view.dataview_selector.select(0)
        self._rebuild_view()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_save(self, _: Any = None) -> bool:
        """Save the collected stats."""
        if self._file_mode:
            self.view.notification.set_text("")
            return False
        model = self.model.active_austin

        def _dump_stats() -> None:
            assert self.model.system.child_process is not None
            pid = self.model.system.child_process.pid
            output_file = Path(f"austin_{int(time())}_{pid}").with_suffix(
                ".mojo"
            )
            try:
                with output_file.open("wb") as stream:
                    mojo_writer = MojoStreamWriter(stream)
                    for k, v in model.metadata.items():
                        mojo_writer.write(AustinMetadata(k, v))
                    for event in model.stats.flatten():
                        mojo_writer.write(event)
                self.view.notification.set_text(
                    self.view.markup(
                        f"Stats saved as <running>{escape(str(output_file))}</running> "
                    )
                )
            except IOError as e:
                self.view.notification.set_text(f"Failed to save stats: {e}")

            self.view.root_widget.refresh()

        await asyncio.get_event_loop().run_in_executor(None, _dump_stats)

        return False

    async def on_play_pause(self, _: Any = None) -> bool:
        """On play/pause handler.

        Pausing seals the current revision in O(1) time (no deep copy) and the
        screen is immediately refreshed from the sealed revision. Resuming is
        requested here and committed atomically by the reducer once the event
        queue is fully drained.
        """
        if self.view._stopped:
            return False

        if self.model.frozen:
            self.model.request_resume()
            self.view.notification.set_text("Resuming ...")
            return True

        self.model.request_freeze()
        # Commit the freeze immediately: seal + rotate is O(1) and events still
        # queued naturally flow into the fresh live segment.
        self.model.reduce(max_events=0)

        # The sealed revision is the live one just sealed. The current view
        # already shows it unless unrendered current-thread data is pending,
        # in which case the screen catches up from the sealed revision.
        sealed = self.model.active_austin
        if self._force_view or (
            sealed.threads
            and sealed.threads[sealed.current_thread] in self._pending_dirty
        ):
            self._render_view()

        self.view.notification.set_text("Paused")
        return True

    def _change_threshold(self, delta: float) -> float:
        austin = self.model.active_austin
        austin.threshold += delta

        if austin.threshold < 0.0:
            austin.threshold = 0.0
        elif austin.threshold > 1.0:
            austin.threshold = 1.0

        if self.view._stopped or self.model.frozen:
            self._force_view = True
            self._rebuild_view()
            self.view.table.draw()
            self.view.table.refresh()

        return austin.threshold

    async def on_threshold_up(self, _: Any = None) -> bool:
        """Handle threshold up."""
        th = self._change_threshold(0.01) * 100.0
        self.view.threshold.set_text(f"{th:.0f}%")
        return True

    async def on_threshold_down(self, _: Any = None) -> bool:
        """Handle threshold down."""
        th = self._change_threshold(-0.01) * 100.0
        self.view.threshold.set_text(f"{th:.0f}%")
        return True

    async def on_graph_selected(self, _: Any = None) -> bool:
        """Select graph visualisation."""
        if self._view_mode is AustinViewMode.GRAPH:
            return False

        self._view_mode = AustinViewMode.GRAPH

        self.view.dataview_selector.select(1)

        self._rebuild_view()

        return True

    def shutdown(self) -> None:
        """Force quit: terminate Austin and close the view immediately."""
        try:
            if self.austin is not None:
                self.austin.terminate()
        except Exception:
            pass
        try:
            self.view.close()
        except Exception:
            pass

    def on_shutdown(self, _: Any = None) -> None:
        """The shutdown view event handler."""
        self.shutdown()

    def on_exception(self, exc: Exception) -> None:
        """The exception view event handler."""
        self.shutdown()
        raise exc

    # Austin events

    async def on_sample(self, sample: AustinSample) -> None:
        """Austin sample received callback.

        Samples are only enqueued; the expensive reduction happens in bounded
        batches from the update loop so that keyboard handling stays
        responsive and no sample is ever dropped.
        """
        self.model.submit(sample)

    async def on_metadata(self, metadata: AustinMetadata) -> None:
        """Austin metadata received callback.

        Metadata is queued together with the samples so that it is always
        applied *before* the samples that depend on it (e.g. the stats type
        configured by the ``mode`` metadata).
        """
        self.model.submit(metadata)

    def _on_metadata_reduced(self, metadata: AustinMetadata) -> None:
        """Apply view/system side effects of metadata in queue order."""
        if metadata.name == "mode":
            self.view.set_mode(metadata.value)
        elif metadata.name == "python":
            self.view.set_python(metadata.value)
        elif metadata.name == "duration":
            self.model.system._duration = int(metadata.value) / 1e6

    async def on_terminate(self) -> None:
        """Austin terminate callback."""
        await self.stop()

    # View events

    def on_view_event(self, event: AustinView.Event, data: Any = None) -> None:
        """View events handler."""

        def _unhandled(_: Any) -> None:
            raise RuntimeError(f"Unhandled view event: {event}")

        {
            AustinView.Event.QUIT: self.on_shutdown,
            AustinView.Event.EXCEPTION: self.on_exception,
        }.get(event, _unhandled)(data)  # type: ignore[operator]
