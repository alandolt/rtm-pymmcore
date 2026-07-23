"""Launch napari-micromanager on a faro microscope class.

``python -m napari_micromanager -c some.cfg`` only knows how to load a
Micro-Manager ``.cfg``. Several Pertzlab scopes need Python-side setup that
cannot live in a cfg file -- the DMD wake/blank loop and hold exposure
(Moench, Niesen), the ROI-follows-binning hook, the laser keepalive, the
per-scope MDA engine. All of that lives in the microscope classes under
:mod:`faro.microscope.pertzlab`.

This launcher builds the microscope object first (so the cfg *and* the
Python-side setup are applied), then hands its already-configured
``CMMCorePlus`` to ``napari_micromanager.MainWindow(viewer, mmcore=...)``.
The plugin drives that core without owning it, so faro's MDA engine stays
registered and teardown remains the microscope's job.

Examples
--------
    uv run faro-napari moench
    uv run faro-napari niesen-nolaser
    uv run faro-napari moench --affine E:/calib/affine.npy
    uv run faro-napari demo
    uv run faro-napari mypkg.scopes:MyScope --kwarg port=3 --kwarg fast=True

Equivalent without a reinstall::

    uv run python -m faro.napari_launcher moench

Inside the napari console, ``mic``, ``mmcore`` and ``viewer`` are available,
so e.g. ``mic.calibrate_dmd(CyanStim)`` can be run interactively.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

# NOTE ON IMPORT ORDER: this module lives inside the ``faro`` package, so
# ``faro/__init__.py`` -- which pins ``PYMM_SIGNALS_BACKEND=psygnal`` -- has
# already run by the time anything here is imported. napari / pymmcore_widgets
# must not be imported before that (pymmcore_widgets sets the backend to 'qt',
# which routes frameReady through Qt's queued delivery and starves faro's
# controller pipeline). Everything Qt-flavoured is therefore imported lazily,
# inside the functions below.


# ---------------------------------------------------------------------------
# Scope registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeSpec:
    """A named, ready-to-run microscope configuration."""

    target: str  # "module.path:ClassName"
    help: str
    kwargs: dict[str, Any] = field(default_factory=dict)


SCOPES: dict[str, ScopeSpec] = {
    "moench": ScopeSpec(
        "faro.microscope.pertzlab.moench:Moench",
        "Moench (Ti + Mosaic3 DMD), cropped ROI, DMD keepalive running.",
    ),
    "moench-uncropped": ScopeSpec(
        "faro.microscope.pertzlab.moench:Moench",
        "Moench with the camera ROI left at full frame.",
        {"uncropped": True},
    ),
    "niesen": ScopeSpec(
        "faro.microscope.pertzlab.niesen:Niesen",
        "Niesen (Ti2 + Polygon1000 DMD) with the laser on and warmed up.",
    ),
    "niesen-fast": ScopeSpec(
        "faro.microscope.pertzlab.niesen:Niesen",
        "Niesen with the laser on but skipping the 10 s warm-up wait.",
        {"fast_init": True},
    ),
    "niesen-nolaser": ScopeSpec(
        "faro.microscope.pertzlab.niesen:Niesen",
        "Niesen on the no-laser cfg (no laser device, no keepalive).",
        {"use_laser": False},
    ),
    "jungfrau": ScopeSpec(
        "faro.microscope.pertzlab.jungfrau:Jungfrau",
        "Jungfrau (TiFluoro + TTL NIDAQ). No DMD is created by the class.",
    ),
    "demo": ScopeSpec(
        "faro.microscope.demo:MMDemo",
        "Micro-Manager demo config -- no hardware, for testing the launcher.",
    ),
}


def _format_scope_table() -> str:
    width = max(len(name) for name in SCOPES)
    lines = [f"  {name.ljust(width)}  {spec.help}" for name, spec in SCOPES.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def resolve_scope(name: str) -> ScopeSpec:
    """Return the :class:`ScopeSpec` for a registry name or ``module:Class``."""
    if name in SCOPES:
        return SCOPES[name]
    if ":" in name:
        return ScopeSpec(name, "user-supplied microscope class")
    raise SystemExit(
        f"Unknown scope {name!r}. Either use a registered name:\n"
        f"{_format_scope_table()}\n"
        "or pass an import path, e.g. 'faro.microscope.demo:MMDemo'."
    )


def import_target(target: str):
    """Import ``"module.path:ClassName"`` and return the class."""
    module_path, _, class_name = target.partition(":")
    if not module_path or not class_name:
        raise SystemExit(
            f"Bad microscope target {target!r}; expected 'module.path:ClassName'."
        )
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise SystemExit(f"Could not import {module_path!r}: {exc}") from exc
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise SystemExit(
            f"Module {module_path!r} has no attribute {class_name!r}."
        ) from None


def parse_kwarg(item: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI pair; the value is Python-literal-parsed.

    ``--kwarg use_laser=False`` yields ``("use_laser", False)`` rather than
    the string ``"False"`` (which is truthy and would silently do the
    opposite of what was asked). Values that aren't valid Python literals
    are kept as plain strings, so paths like ``C:\\foo\\bar.cfg`` work
    unquoted.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise SystemExit(f"--kwarg expects key=value, got {item!r}.")
    key = key.strip()
    try:
        return key, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return key, raw


def _check_kwargs(cls: type, kwargs: dict[str, Any]) -> None:
    """Fail loudly on kwargs the microscope's ``__init__`` won't accept.

    Silently dropping them would start the scope in a state the user did
    not ask for (e.g. a typo'd ``use_lasr=False`` leaving the laser on).
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return  # no introspectable signature; let the call itself fail
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    accepted = {n for n, p in params.items() if n != "self"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise SystemExit(
            f"{cls.__name__}.__init__ does not accept {', '.join(unknown)}. "
            f"Accepted: {', '.join(sorted(accepted)) or '(none)'}."
        )


def _accepts(cls: type, name: str) -> bool:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return name in params


def build_microscope(
    scope: str,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    config: str | None = None,
    affine: str | None = None,
):
    """Instantiate the requested microscope, fully initialised.

    Returns the microscope instance. Construction runs the class's own
    ``init_scope()``, so the Micro-Manager cfg is loaded, the per-scope MDA
    engine is registered, and DMD/laser/ROI hooks are live before napari
    ever sees the core.
    """
    spec = resolve_scope(scope)
    cls = import_target(spec.target)

    kwargs: dict[str, Any] = dict(spec.kwargs)
    kwargs.update(extra_kwargs or {})

    affine_matrix = None
    if affine is not None:
        import numpy as np

        affine_matrix = np.load(affine)
        if _accepts(cls, "affine_calibration_matrix"):
            kwargs.setdefault("affine_calibration_matrix", affine_matrix)

    if config is not None:
        # init_scope() reads MICROMANAGER_CONFIG off the instance, which
        # resolves to the class attribute -- so overriding it on the class
        # before construction is the only hook available.
        if not hasattr(cls, "MICROMANAGER_CONFIG"):
            raise SystemExit(
                f"--config given but {cls.__name__} has no MICROMANAGER_CONFIG "
                "attribute to override."
            )
        cls.MICROMANAGER_CONFIG = config

    _check_kwargs(cls, kwargs)

    pretty = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"[faro] building {cls.__name__}({pretty})", flush=True)
    mic = cls(**kwargs)

    # Scopes that build their DMD outside __init__ (or don't take an affine
    # kwarg at all) still get the matrix applied here.
    if affine_matrix is not None and "affine_calibration_matrix" not in kwargs:
        dmd = getattr(mic, "dmd", None)
        if dmd is None:
            warnings.warn(
                f"--affine was given but {cls.__name__} has no DMD; the "
                "calibration matrix was not applied.",
                stacklevel=2,
            )
        else:
            dmd.affine = affine_matrix

    return mic


def _adopt_as_singleton(core) -> None:
    """Make *core* the ``CMMCorePlus.instance()`` singleton.

    pymmcore-widgets constructed without an explicit ``mmcore=`` fall back
    to the singleton. The launcher builds the microscope before napari, so
    its core is normally the first one created and already holds the slot --
    but if anything else got there first, a widget opened from the napari
    menu would silently drive a *different*, unconfigured core. Pin it.
    """
    import weakref

    try:
        from pymmcore_plus.core import _mmcore_plus

        _mmcore_plus._instance = weakref.ref(core)
    except Exception:  # pragma: no cover - private API moved upstream
        warnings.warn(
            "Could not pin the microscope's core as CMMCorePlus.instance(); "
            "widgets created without an explicit mmcore= may attach to a "
            "different core.",
            stacklevel=2,
        )


def _push_to_console(viewer, namespace: dict[str, Any]) -> None:
    """Expose objects in napari's embedded IPython console, if present."""
    try:
        console = getattr(viewer.window._qt_viewer, "console", None)
    except Exception:  # pragma: no cover - napari internals moved
        console = None
    if console is not None:
        console.push(namespace)


def _describe(mic) -> None:
    """Print a short summary of what the GUI is now driving."""
    mmc = getattr(mic, "mmc", None)
    print(f"[faro] microscope : {type(mic).__name__}")
    cfg = getattr(mic, "MICROMANAGER_CONFIG", None) or getattr(
        mic, "micromanager_config", None
    )
    if cfg:
        print(f"[faro] cfg        : {cfg}")
    if mmc is not None:
        engine = type(mmc.mda.engine).__name__ if mmc.mda.engine else "(default)"
        print(f"[faro] mda engine : {engine}")
        print(f"[faro] signals    : {type(mmc.mda.events).__name__}")
    dmd = getattr(mic, "dmd", None)
    if dmd is not None:
        state = "calibrated" if getattr(dmd, "affine", None) is not None else (
            "NOT calibrated -- run mic.calibrate_dmd(<channel>) in the console"
        )
        print(f"[faro] dmd        : {state}")
    print("[faro] console    : mic, mmcore, viewer")


def _shutdown(mic, core) -> None:
    """Release the microscope's hardware once the viewer has closed.

    Two things happen here that are easy to get wrong:

    *Signals stay blocked from here on.* ``unloadAllDevices()`` re-emits
    ``systemConfigurationLoaded`` (and friends). By this point the napari
    window is gone, but the pymmcore-widgets that lived in its toolbars are
    still connected to those core signals, so each callback runs against a
    deleted C++ object and pymmcore-plus logs a full traceback per callback
    ("wrapped C/C++ object of type QComboBox has been deleted"). The block
    is deliberately *not* scoped to a context manager: teardown also runs
    again from ``PyMMCoreMicroscope``'s atexit hook, after this function has
    returned, and that pass would re-raise the same noise. Nothing is left
    to listen for once the GUI is down.

    *Both teardown entry points are called.* ``shutdown()`` is only
    overridden on the scopes with extra state to unwind (Moench, Niesen);
    on the others it inherits ``AbstractMicroscope``'s no-op, and the actual
    device unload would be left to the atexit hook. Calling
    ``_teardown_hardware()`` as well makes the unload happen here, while we
    can still report a failure. Both are idempotent and swallow device
    errors internally, so the atexit re-run is harmless.
    """
    print("[faro] shutting down microscope...", flush=True)

    # block() is psygnal's; the Qt signaler has no equivalent, so a missing
    # method just means the noise isn't suppressed -- not a failure.
    block = getattr(getattr(core, "events", None), "block", None)
    if callable(block):
        try:
            block()
        except Exception:  # noqa: BLE001  # pragma: no cover
            pass

    for name in ("shutdown", "_teardown_hardware"):
        method = getattr(mic, name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception as exc:  # noqa: BLE001
            print(f"[faro] {name}() failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faro-napari",
        description=(
            "Run napari-micromanager against a faro microscope class, so "
            "Python-side setup (DMD blanking/hold, ROI hooks, laser "
            "keepalive, per-scope MDA engine) is applied -- not just the "
            "Micro-Manager cfg."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Registered scopes:\n" + _format_scope_table(),
    )
    parser.add_argument(
        "scope",
        nargs="?",
        help="registered scope name, or an import path 'module.path:ClassName'",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the registered scopes and exit",
    )
    parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra keyword argument for the microscope constructor "
            "(repeatable); values are parsed as Python literals"
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="override the class's MICROMANAGER_CONFIG cfg path",
    )
    parser.add_argument(
        "--affine",
        metavar="PATH.npy",
        help="load a saved DMD affine calibration matrix (np.load)",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help=(
            "skip mic.shutdown() when the viewer closes (devices stay "
            "loaded; useful when debugging teardown)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        print("Registered scopes:\n" + _format_scope_table())
        return
    if not args.scope:
        parser.error("a scope is required (see --list)")

    extra = dict(parse_kwarg(item) for item in args.kwarg)

    mic = build_microscope(
        args.scope,
        extra_kwargs=extra,
        config=args.config,
        affine=args.affine,
    )

    core = getattr(mic, "mmc", None)
    if core is None:
        raise SystemExit(
            f"{type(mic).__name__} exposes no .mmc core; napari-micromanager "
            "needs a CMMCorePlus instance."
        )
    _adopt_as_singleton(core)

    # Imported only now: pymmcore_widgets (pulled in transitively) forces
    # PYMM_SIGNALS_BACKEND='qt' at import, which must not happen before the
    # microscope's MDA runner exists.
    import napari

    from napari_micromanager.main_window import MainWindow

    viewer = napari.Viewer()
    # mmcore= means the plugin drives this core without owning it: it will
    # not cancel MDAs or unload devices on close, leaving teardown to the
    # microscope (and to the finally block below).
    win = MainWindow(viewer, mmcore=core)
    dock = viewer.window.add_dock_widget(win, name="MicroManager", area="top")
    if hasattr(dock, "_close_btn"):
        dock._close_btn = False

    _push_to_console(viewer, {"mic": mic, "mmcore": core, "viewer": viewer})
    _describe(mic)

    try:
        napari.run()
    finally:
        if not args.no_shutdown:
            _shutdown(mic, core)


if __name__ == "__main__":
    main()
