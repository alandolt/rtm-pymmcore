import logging
import threading
import time
import weakref
from typing import Optional

import requests
import pymmcore_plus

# Disable pymmcore-plus's rotating file log handler at import time. On Windows
# its RotatingFileHandler crashes on rollover — os.rename() raises
# PermissionError (WinError 32) because the .log file is still held open by
# this (or another) process. file=None drops the file handler entirely so no
# rollover is ever attempted; stderr_level="CRITICAL" keeps the console quiet.
# Mirrors the Jungfrau microscope. Must run before CMMCorePlus is constructed.
pymmcore_plus.configure_logging(file=None, stderr_level="CRITICAL")

from pymmcore_plus.mda._engine import MDAEngine
from useq import MDAEvent

from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.core.dmd import DMD
from faro.core.data_structures import ImgType
from faro.core._useq_compat import SLMImage


class WakeUpLaser:
    def __init__(self, lumencore_ip="192.168.201.200"):
        self.ip = lumencore_ip
        self.last_wakeup = 0.0
        self.thread: threading.Thread | None = None
        # daemon=True so interpreter shutdown doesn't block on this thread
        # (leaving a zombie python.exe that holds the next session hostage);
        # the Event lets stop() break out of the sleep immediately.
        self._stop_event = threading.Event()

    def wakeup_laser(self):
        url = f"http://{self.ip}/service/?command=WAKEUP"
        requests.get(url, timeout=5)

    def run(self, wait_for_warmup=True):
        # Idempotent: a keepalive already running stays as-is.
        if self.thread is not None and self.thread.is_alive():
            return
        self._stop_event.clear()
        self.last_wakeup = 0.0
        self.thread = threading.Thread(target=self._keep_alive, daemon=True)
        self.thread.start()
        if wait_for_warmup:
            time.sleep(15)

    def _keep_alive(self):
        while not self._stop_event.is_set():
            if time.time() - self.last_wakeup > 60:
                self.wakeup_laser()
                self.last_wakeup = time.time()
            # Event.wait lets stop() return promptly instead of eating up
            # to 3 s of teardown time.
            if self._stop_event.wait(timeout=3):
                return

    def stop(self):
        self._stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = None


class Niesen(PyMMCoreMicroscope):
    MICROMANAGER_PATH = r"C:\Program Files\Micro-Manager-2.0_api75"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\Niesen\\Ti2Niesen.cfg"
    # Config loaded when the microscope is constructed with use_laser=False:
    # a variant without the laser device so the system starts up without the
    # laser hardware present (and without the WakeUpLaser keepalive).
    MICROMANAGER_CONFIG_NO_LASER = (
        "E:\\pertzlab_mic_configs\\micromanager\\Niesen\\Ti2Niesen_noLaser.cfg"
    )
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False
    DMD_CHANNEL_GROUP = "TTL_ERK"
    # The imaging light path is LED -> DMD -> sample for *every* channel, so the
    # DMD must be held full-open (all-on) during each imaging frame, not just on
    # stim frames. The controller only attaches an SLM image to stim events;
    # imaging events leave the DMD latched at whatever the last frame displayed,
    # so after a stim frame the next imaging frame can come back dark. With this
    # flag, NiesenMDAEngine injects an all-on SLMImage on every non-stim frame
    # (mirrors Moench's DMD_NEEDS_TO_BE_WAKEN behaviour).
    DMD_NEEDS_TO_BE_WAKEN = True
    # Channels whose imaging light path runs through the DMD
    # (LED -> DMD -> sample). Only for these does the DMD need to be held
    # full-open (all-on) on the imaging frame — mirroring the Moench, where
    # every channel images through the DMD. Other Niesen channels image on a
    # path that bypasses the DMD, so forcing the DMD open for them is wrong.
    #
    # Map each channel group to the presets within it that image through the
    # DMD: {channel_group: (preset, preset, ...)}. When this mapping is empty,
    # the engine falls back to waking the DMD on *every* non-stim frame
    # (the previous behaviour).
    IMAGE_THROUGH_DMD: dict[str, tuple[str, ...]] = {
        "TTL_ERK": ("CyanStim", "mScarlet3", "miRFP"),
    }

    # --- Mightex Polygon 1000 DMD hold / settle ---
    # The Polygon 1000, like the Moench's Mosaic3, has no indefinite "mirror on"
    # hold: after a ``displaySLMImage`` the micromirrors hold the pattern only
    # for the SLM ``ExposureTime`` and then *park*. A short exposure therefore
    # blanks the DMD mid-frame -- exactly the "blank capture" already documented
    # in ``DMD.calibrate`` (OverlapMode=Off) and the reason ``display_livemode``
    # forces a 200 s exposure. During an MDA we force ``DMD_HOLD_EXPOSURE_MS`` on
    # every displayed pattern (stim mask or injected all-on) so the mirrors stay
    # put across the whole camera window, and pause ``DMD_SETTLE_MS`` between the
    # mirror commit and the snap so the camera never integrates against a
    # not-yet-committed pattern.
    #
    # Dose is unaffected: illumination is gated in *time* by the NIDAQ AO
    # blanking task triggered by the camera exposure-out line (see the config
    # header), not by the DMD, so holding the pattern longer delivers no extra
    # light. 200000 ms = 200 s matches ``DMD.display_livemode``. Set to None/0
    # to disable either override.
    DMD_HOLD_EXPOSURE_MS: float = 200000.0
    DMD_SETTLE_MS: float = 10.0
    POWER_PROPERTIES = {
        "CyanStim": ("LedDMD", "Cyan_Level"),
        "mScarlet3": ("LedDMD", "Green_Level"),
        "miRFP": ("Laser", "RED_Intensity"),
        "mCitrine": ("Laser", "CYAN_Intensity")

    }
    # --- Camera ROI: crop the field down to the DMD-illuminated area ---
    # Binning is NOT pinned here: it's set by the per-experiment channel
    # presets (e.g. the TTL_ERK group). The ROI_* values below are in
    # *binned* pixels and are written for REFERENCE_BINNING; set_roi()
    # reads the camera's actual binning and rescales the crop to it, so the
    # same ROI_* numbers work whatever binning the experiment config selects.
    SET_ROI_REQUIRED = True
    REFERENCE_BINNING = "2x2"
    ROI_X = 0
    ROI_Y = 60
    ROI_WIDTH = 1024
    ROI_HEIGHT = 1000

    # Whether stim on this rig projects a *spatial* pattern that has to be
    # mapped from camera space into DMD space via the DMD affine matrix.
    # The current Niesen stim path is whole-FOV: the stimulator hands the DMD
    # the scalar ``True`` sentinel, which is displayed all-on without an
    # affine transform, so the "DMD not calibrated" check in
    # ``AbstractMicroscope._validate_dmd_calibration`` would fire on every run
    # for nothing. Set this back to True (per class or per instance) for
    # patterned experiments so the missing-calibration warning is restored.
    STIM_USES_DMD_PATTERN = False

    def __init__(self, affine_calibration_matrix=None, fast_init=False, use_laser=True):
        super().__init__()
        # When use_laser=False, run without the laser: load the no-laser
        # Micro-Manager config and skip the WakeUpLaser device/keepalive
        # entirely (self.wl stays None, so every laser hook below no-ops).
        self.use_laser = use_laser
        if not use_laser:
            self.MICROMANAGER_CONFIG = self.MICROMANAGER_CONFIG_NO_LASER
        self.dmd_needs_to_be_waken = self.DMD_NEEDS_TO_BE_WAKEN
        # Re-assert file=None in case something re-enabled the file handler
        # after import (mirrors Jungfrau). See the module-level call above.
        pymmcore_plus.configure_logging(file=None, stderr_level="CRITICAL")
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        if use_laser:
            self.wl = WakeUpLaser()
            self.wl.wakeup_laser()
            if not fast_init:
                time.sleep(10)
        else:
            self.wl = None
        self.init_scope()
        # The laser keepalive is NOT started here: we don't want it pinging
        # during construction or the idle waiting phase before a run. It is
        # started on demand the moment imaging begins — GUI live view (via
        # the continuousSequenceAcquisitionStarted hook wired in init_scope)
        # or the controller taking over (run_mda) — and stopped again in
        # post_experiment() when the experiment finishes.
        self.dmd = DMD(
            self.mmc,
            resolve_power=self.resolve_power,
            affine_matrix=affine_calibration_matrix,
        )
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        if self.wl is not None:
            self.wl.wakeup_laser()
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.mmc.setSLMPixelsTo(self.slm_dev, 255)
        self.mmc.displaySLMImage(self.slm_dev)
        self.mmc.setChannelGroup(channelGroup=self.DMD_CHANNEL_GROUP)
        # Start the laser keepalive the moment GUI live view begins.
        # napari-micromanager's live button starts a continuous sequence,
        # which fires this signal; psygnal holds the bound method weakly so
        # this doesn't pin the microscope alive.
        self.mmc.events.continuousSequenceAcquisitionStarted.connect(
            self._on_live_view_started
        )
        # Keep the camera cropped to the DMD ROI across binning changes.
        # Binning isn't pinned here (the per-experiment channel presets set
        # it), and MM drivers reset the ROI whenever binning changes, so we
        # re-apply the crop on every binning change: configSet covers presets
        # applied via setConfig (including mid-MDA), propertyChanged covers a
        # direct binning set (e.g. the GUI dropdown). psygnal holds these
        # bound methods weakly, so they don't pin the microscope alive.
        if self.SET_ROI_REQUIRED:
            self._last_roi_binning = None
            self.mmc.events.configSet.connect(self._sync_roi_to_binning)
            self.mmc.events.propertyChanged.connect(self._on_property_changed_roi)
            # Crop for the binning that's active right now (e.g. live view
            # before any experiment preset has run).
            self._sync_roi_to_binning()
        # Register the Niesen MDA engine so imaging frames arm the DMD all-on
        # (LED -> DMD -> sample path). See DMD_NEEDS_TO_BE_WAKEN above.
        self.register_engine()

    @staticmethod
    def _binning_factor(binning) -> int:
        """Parse a Micro-Manager binning value ('2x2', '2', '1x1') -> int factor.

        Falls back to 1 (no binning) if the value can't be parsed.
        """
        token = str(binning).lower().split("x")[0].strip()
        try:
            return max(1, int(token))
        except ValueError:
            return 1

    def set_roi(self):
        """Crop the camera to the DMD area, rescaled to the current binning.

        ``ROI_*`` are defined in binned pixels at ``REFERENCE_BINNING``.
        Binned-pixel counts scale inversely with the binning factor, so the
        same physical crop at a different binning is
        ``ROI * (reference_factor / current_factor)``. Reading the camera's
        actual binning means a binning change is followed automatically — no
        manual ROI recomputation. Re-call this after changing binning at
        runtime (the camera resets its ROI on a binning change).
        """
        self.mmc.clearROI()
        ref_factor = self._binning_factor(self.REFERENCE_BINNING)
        try:
            cur_binning = self.mmc.getProperty(
                self.mmc.getCameraDevice(), "Binning"
            )
        except Exception:
            cur_binning = self.REFERENCE_BINNING
        cur_factor = self._binning_factor(cur_binning)
        scale = ref_factor / cur_factor
        self.mmc.setROI(
            round(self.ROI_X * scale),
            round(self.ROI_Y * scale),
            round(self.ROI_WIDTH * scale),
            round(self.ROI_HEIGHT * scale),
        )

    def _sync_roi_to_binning(self, *args) -> None:
        """Re-crop to the DMD ROI whenever the camera binning changes.

        Binning is not pinned here — the per-experiment channel presets (e.g.
        the TTL_ERK group) set it — and MM camera drivers reset the ROI to
        full-frame on every binning change. So the crop must be re-applied
        each time binning changes. This is connected to ``configSet`` (which
        fires when a preset is applied via ``setConfig``, including mid-MDA
        via ``_set_event_channel``) and, through ``_on_property_changed_roi``,
        to a direct binning change (e.g. the GUI dropdown). The
        ``_last_roi_binning`` guard makes it a no-op when binning is unchanged,
        so it's cheap to leave wired to every ``configSet``.

        Accepts ``*args`` so it can serve both as the ``configSet`` slot
        (``group, config``) and be called directly with no arguments.
        """
        if not self.SET_ROI_REQUIRED:
            return
        try:
            cur_binning = self.mmc.getProperty(
                self.mmc.getCameraDevice(), "Binning"
            )
        except Exception:
            return
        if cur_binning == getattr(self, "_last_roi_binning", None):
            return
        self._last_roi_binning = cur_binning
        try:
            self.set_roi()
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to re-apply ROI after binning change to %s",
                cur_binning,
                exc_info=True,
            )

    def _on_property_changed_roi(self, device: str, prop: str, value: str) -> None:
        """Re-crop when the camera ``Binning`` property is set directly.

        Covers a binning change that arrives as a bare ``setProperty`` (e.g.
        the napari-micromanager binning dropdown) rather than a ``setConfig``.
        """
        if prop == "Binning":
            self._sync_roi_to_binning()

    def calibrate_dmd(
        self,
        calibration_channel,
        verbose=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
    ):
        """Calibrate the DMD. Always runs the calibration when called."""
        if self.dmd is not None:
            self.dmd.calibrate(
                calibration_channel,
                verbose=verbose,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )

    def _validate_dmd_calibration(self, events) -> bool:
        """Skip the missing-calibration warning for pattern-free stim.

        See :attr:`STIM_USES_DMD_PATTERN`: whole-FOV stim drives the DMD with
        the scalar ``True`` sentinel, which never goes through
        ``DMD.affine_transform``, so an uncalibrated DMD is not an error here.
        Kept local to this rig so the base check stays strict everywhere else.
        """
        if not self.STIM_USES_DMD_PATTERN:
            return True
        return super()._validate_dmd_calibration(events)

    def register_engine(self, force: bool = False) -> None:
        """Create and register the Niesen-specific MDA engine.

        Idempotent unless ``force=True``. Attaches a weakref to this
        microscope on the engine (so the engine can read ``dmd`` /
        ``dmd_needs_to_be_waken`` at event time) and registers it on
        ``self.mmc.mda``.
        """
        if getattr(self, "engine", None) is not None and not force:
            return

        self.engine = NiesenMDAEngine(self.mmc)
        try:
            self.engine.attach_microscope(self)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to attach microscope to engine"
            )
        try:
            self.mmc.mda.set_engine(self.engine)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to register MDA engine on mmc.mda"
            )

    def run_mda(self, event_iter):
        """Start the laser keepalive before the controller's acquisition.

        This is the "controller takes over" trigger. ``wait_for_warmup``
        blocks ~15 s on the *first* start so the first frame isn't dark;
        :meth:`WakeUpLaser.run` is idempotent, so later batches — and the
        case where GUI live view already started the keepalive — don't wait
        again.
        """
        self._ensure_laser_running(wait_for_warmup=True)
        return super().run_mda(event_iter)

    def _on_live_view_started(self, *args):
        """GUI live view started -> keep the laser alive.

        Runs on the acquisition-start signal, so it must NOT block on
        warmup (that would freeze the GUI). The keepalive pings the laser
        on its first loop iteration regardless.
        """
        self._ensure_laser_running(wait_for_warmup=False)

    def _ensure_laser_running(self, wait_for_warmup=False):
        """Start the laser keepalive if it isn't already running (idempotent)."""
        wl = getattr(self, "wl", None)
        if wl is None:
            return
        try:
            wl.run(wait_for_warmup=wait_for_warmup)
        except Exception:
            pass

    def post_experiment(self):
        """Stop the laser keepalive when the experiment finishes.

        Called by ``Controller.finish_experiment()``. The keepalive is
        restarted on demand by :meth:`run_mda` / GUI live view for the next
        run, so the laser isn't pinged during the idle phase between
        experiments.
        """
        wl = getattr(self, "wl", None)
        if wl is not None:
            try:
                wl.stop()
            except Exception:
                pass

    def shutdown(self):
        """Tear down hardware state so the microscope can be discarded.

        Delegates to :meth:`_teardown_hardware`, which is also what the
        atexit hook registered in :class:`PyMMCoreMicroscope` calls — so
        explicit shutdown and interpreter-exit cleanup run identically.
        """
        self._teardown_hardware()

    def _teardown_hardware(self) -> None:
        """Stop the laser keepalive thread, then delegate to base teardown.

        Overriding this (rather than only ``shutdown``) is what makes the
        atexit hook stop the laser thread: the hook calls
        ``_teardown_hardware``, not ``shutdown``. The base teardown then
        cancels any running MDA and unloads all Micro-Manager devices so
        COM ports and the SLM handle are released — without it, pymmcore's
        native threads keep the process alive after the main thread exits,
        leaving a zombie that blocks the next session.
        """
        wl = getattr(self, "wl", None)
        if wl is not None:
            try:
                wl.stop()
            except Exception:
                pass
        super()._teardown_hardware()


class NiesenMDAEngine(MDAEngine):
    """MDA engine for Niesen: arm the DMD all-on for every imaging frame.

    The Niesen imaging path is LED -> DMD -> sample for *all* channels, so
    the DMD must be full-open during each imaging snap. The controller only
    attaches an SLM image to stim events; imaging events carry
    ``slm_image=None``, which leaves the DMD latched at whatever the previous
    frame displayed. After a (whole-FOV) stim frame the DMD can blank or hold
    the stim pattern, so the next imaging frame would come back dark.

    This engine injects an all-on ``SLMImage`` on every non-stim frame so the
    base engine's ``_set_event_slm_image`` / ``_exec_event_slm_image`` re-arm
    the DMD full-open. It mirrors Moench's ``_maybe_inject_dmd_wake_slm`` but
    keeps the rest of the base engine untouched (Niesen has no stuck-Busy
    stage and no DMD keep-alive thread to pause).
    """

    def __init__(
        self,
        mmc,
        *,
        use_hardware_sequencing: bool = False,
        restore_initial_state: Optional[bool] = None,
    ):
        super().__init__(
            mmc,
            use_hardware_sequencing=use_hardware_sequencing,
            restore_initial_state=restore_initial_state,
        )
        self._microscope_ref: Optional[weakref.ref] = None
        self._log = logging.getLogger(self.__class__.__name__)

    def attach_microscope(self, mic) -> None:
        """Attach the microscope instance (weakref) so the engine can consult it."""
        self._microscope_ref = weakref.ref(mic)

    @property
    def microscope(self):
        return None if self._microscope_ref is None else self._microscope_ref()

    def setup_event(self, event: MDAEvent) -> None:
        # Inject before the base engine sets up the SLM for this snap.
        super().setup_event(self._maybe_inject_dmd_wake_slm(event))

    def exec_event(self, event: MDAEvent):
        # Must mirror setup_event: the MDARunner keeps its own reference to
        # the original event, so the model_copy inside setup_event doesn't
        # propagate here. Without re-injecting, _exec_event_slm_image (the
        # displaySLMImage that arms the pattern for the camera TTL) never
        # fires for non-stim frames and the DMD isn't re-armed all-on.
        yield from super().exec_event(self._maybe_inject_dmd_wake_slm(event))

    @staticmethod
    def _event_images_through_dmd(mic, event: MDAEvent) -> bool:
        """Whether *event*'s channel images through the DMD (needs waking).

        Consults the microscope's ``IMAGE_THROUGH_DMD`` mapping. When that
        mapping is empty (or absent), every non-stim frame is treated as a
        DMD imaging frame — the previous, channel-agnostic behaviour.
        Otherwise only channels whose (group, preset) appears in the mapping
        are woken; all other channels image on a path that bypasses the DMD.
        """
        mapping = getattr(mic, "IMAGE_THROUGH_DMD", None)
        if not mapping:
            return True
        ch = event.channel
        if ch is None:
            return False
        return ch.config in mapping.get(ch.group, ())

    def _maybe_inject_dmd_wake_slm(self, event: MDAEvent) -> MDAEvent:
        """Return *event* with an all-on SLMImage if it's a DMD imaging frame.

        No-op (returns the event unchanged) when: it already carries an SLM
        image (stim frames, DMD calibration events), the microscope doesn't
        want DMD waking, there's no DMD, the event is a stim emission, or the
        event's channel images on a path that bypasses the DMD (see
        ``IMAGE_THROUGH_DMD``).
        """
        if event.slm_image is not None:
            return event
        mic = self.microscope
        if mic is None or not getattr(mic, "dmd_needs_to_be_waken", False):
            return event
        dmd = getattr(mic, "dmd", None)
        if dmd is None:
            return event
        if (event.metadata or {}).get("img_type") == ImgType.IMG_STIM:
            return event
        if not self._event_images_through_dmd(mic, event):
            return event
        # No exposure on the injected SLMImage: the DMD hold time is forced to
        # ``DMD_HOLD_EXPOSURE_MS`` in ``_set_event_slm_image``, and the imaging
        # dose is set by the light source (LED -> DMD -> sample), gated by the
        # camera exposure-out trigger -- not by the DMD hold.
        return event.model_copy(
            update={"slm_image": SLMImage(data=True, device=dmd.name)}
        )

    def _set_event_slm_image(self, event: MDAEvent) -> None:
        """Upload the SLM pattern, then force a long *hold* exposure on the DMD.

        The base method uploads the image and, if the ``SLMImage`` carries an
        exposure, writes it via ``setSLMExposure``. On the Polygon 1000 that
        value is how long the micromirrors hold the pattern after "Expose"
        before they park, so we override it to ``Niesen.DMD_HOLD_EXPOSURE_MS``
        (indefinite in practice) so the mirrors stay in the pattern across the
        whole camera window. The stim *dose* is unaffected -- it is gated by
        the camera-triggered NIDAQ blanking, not the DMD. See the
        ``DMD_HOLD_EXPOSURE_MS`` note on :class:`Niesen`.
        """
        super()._set_event_slm_image(event)
        if event.slm_image is None:
            return
        mic = self.microscope
        hold_ms = (
            getattr(mic, "DMD_HOLD_EXPOSURE_MS", None) if mic is not None else None
        )
        if not hold_ms:
            return
        core = self.mmcore
        slm_device = event.slm_image.device or core.getSLMDevice()
        if not slm_device:
            return
        try:
            core.setSLMExposure(slm_device, float(hold_ms))
        except Exception as e:
            self._log.warning("Failed to set DMD hold exposure. %s", e)

    def _exec_event_slm_image(self, img) -> None:
        """Display the pattern, then settle before the camera snaps.

        ``displaySLMImage`` ("Expose") commits the mask to the micromirrors; the
        short settle lets that commit finish before ``snapImage`` opens the
        camera (and the camera exposure-out line gates the LED on), so the first
        slice of the frame isn't integrated against a not-yet-committed pattern.
        Gated by ``Niesen.DMD_SETTLE_MS`` (None/0 disables).
        """
        super()._exec_event_slm_image(img)
        mic = self.microscope
        settle_ms = getattr(mic, "DMD_SETTLE_MS", None) if mic is not None else None
        if settle_ms:
            time.sleep(float(settle_ms) / 1000.0)
