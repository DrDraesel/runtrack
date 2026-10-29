"""Tests for the ONNX Runtime provider switch (config.ORT_PROVIDER).

No GPU, no model download and no real onnxruntime session: provider resolution
is exercised against explicit ``available`` provider lists, and the RTMPose
backend runs against a fake ``rtmlib`` module, so "DirectML is not available
here -> fall back to CPU" is proven deterministically on any machine (also on a
GPU-less CI box).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import config  # noqa: E402
import pose_backends as PB  # noqa: E402

DML = "DmlExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
#: A machine with every provider (what the resolution rules are tested against).
ALL = (DML, CUDA, CPU)
#: What the app venv exposes today: plain onnxruntime, no GPU provider.
CPU_ONLY = ("AzureExecutionProvider", CPU)
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# --------------------------------------------------------------------- helpers

def _fake_rtmlib():
    """A stand-in ``rtmlib`` whose BodyWithFeet returns a centred HALPE pose."""
    mod = types.ModuleType("rtmlib")

    class BodyWithFeet:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            BodyWithFeet.instances.append(self)

        def __call__(self, image):
            h, w = image.shape[:2]
            k = np.zeros((PB.HALPE26_COUNT, 2), dtype=np.float32)
            k[:] = (0.5 * w, 0.5 * h)
            return k[None], np.full((1, PB.HALPE26_COUNT), 0.9, np.float32)

    mod.BodyWithFeet = BodyWithFeet
    return mod


class _FakeCase(unittest.TestCase):
    """Injects the fake rtmlib for the duration of one test."""

    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {"rtmlib": _fake_rtmlib()})
        patcher.start()
        self.addCleanup(patcher.stop)


# ---------------------------------------------------------------- config key

class TestConfigKey(unittest.TestCase):
    def test_config_exposes_an_ort_provider_setting(self):
        self.assertTrue(hasattr(config, "ORT_PROVIDER"))
        self.assertIsInstance(config.ORT_PROVIDER, str)

    def _load_config_copy(self):
        """Execute config.py as a fresh module (no reload of the imported one)."""
        spec = importlib.util.spec_from_file_location(
            "config_provider_probe", BASE / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_env_var_sets_the_provider(self):
        with mock.patch.dict(os.environ, {"RUNTRACK_ORT_PROVIDER": "directml"}):
            self.assertEqual(self._load_config_copy().ORT_PROVIDER, "directml")

    def test_default_is_auto_when_the_env_var_is_unset(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("RUNTRACK_ORT_PROVIDER", None)
            self.assertEqual(self._load_config_copy().ORT_PROVIDER, "auto")


# ----------------------------------------------------------- provider rules

class TestResolveProvider(unittest.TestCase):
    def test_auto_prefers_directml_over_cuda(self):
        got = PB.resolve_provider("auto", available=ALL)
        self.assertEqual(got.provider, DML)
        self.assertFalse(got.degraded)
        self.assertEqual(got.reason, "")

    def test_auto_uses_cuda_when_directml_is_absent(self):
        got = PB.resolve_provider("auto", available=(CUDA, CPU))
        self.assertEqual(got.provider, CUDA)
        self.assertFalse(got.degraded)

    def test_auto_is_cpu_on_a_cpu_only_onnxruntime(self):
        got = PB.resolve_provider("auto", available=CPU_ONLY)
        self.assertEqual(got.provider, CPU)
        self.assertFalse(got.degraded)         # auto -> CPU is not a degradation
        self.assertIn("CPU", got.reason)

    def test_auto_is_cpu_when_onnxruntime_is_missing(self):
        got = PB.resolve_provider("auto", available=())
        self.assertEqual(got.provider, CPU)

    def test_cpu_is_always_cpu(self):
        self.assertEqual(PB.resolve_provider("cpu", available=ALL).provider, CPU)
        self.assertEqual(PB.resolve_provider("cpu", available=()).provider, CPU)

    def test_directml_is_used_when_available(self):
        got = PB.resolve_provider("directml", available=ALL)
        self.assertEqual(got.provider, DML)
        self.assertFalse(got.degraded)

    def test_unavailable_directml_degrades_to_cpu_with_a_reason(self):
        got = PB.resolve_provider("directml", available=CPU_ONLY)
        self.assertEqual(got.provider, CPU)
        self.assertTrue(got.degraded)
        self.assertIn(DML, got.reason)

    def test_unavailable_cuda_degrades_to_cpu_with_a_reason(self):
        got = PB.resolve_provider("cuda", available=CPU_ONLY)
        self.assertEqual(got.provider, CPU)
        self.assertTrue(got.degraded)
        self.assertIn(CUDA, got.reason)

    def test_unknown_value_falls_back_to_auto(self):
        got = PB.resolve_provider("gpu", available=ALL)
        self.assertEqual(got.provider, DML)      # what auto picks here
        self.assertEqual(got.requested, "gpu")   # the raw request is kept
        self.assertIn("unknown", got.reason.lower())
        self.assertEqual(PB.resolve_provider("gpu", available=CPU_ONLY).provider, CPU)

    def test_value_is_case_and_space_insensitive(self):
        self.assertEqual(PB.resolve_provider(" DirectML ", available=ALL).provider, DML)
        self.assertEqual(PB.resolve_provider("CPU", available=ALL).provider, CPU)

    def test_none_reads_config(self):
        with mock.patch.object(config, "ORT_PROVIDER", "cuda"):
            got = PB.resolve_provider(available=ALL)
        self.assertEqual(got.requested, "cuda")
        self.assertEqual(got.provider, CUDA)

    def test_env_var_is_the_fallback_when_config_is_absent(self):
        with mock.patch.object(config, "ORT_PROVIDER", None):
            with mock.patch.dict(os.environ, {"RUNTRACK_ORT_PROVIDER": "cpu"}):
                self.assertEqual(PB.provider_setting(), "cpu")

    def test_provider_setting_never_raises(self):
        self.assertIsInstance(PB.provider_setting(), str)
        self.assertIsInstance(PB.ort_available_providers(), tuple)


class TestSessionProviders(unittest.TestCase):
    def test_cpu_alone_gets_no_fallback_entry(self):
        self.assertEqual(PB.session_providers(PB.ProviderChoice(CPU, "cpu", False, "")),
                         [CPU])

    def test_gpu_provider_keeps_cpu_as_the_session_fallback(self):
        choice = PB.ProviderChoice(DML, "directml", False, "")
        self.assertEqual(PB.session_providers(choice), [DML, CPU])

    def test_accepts_a_bare_provider_name(self):
        self.assertEqual(PB.session_providers(CUDA), [CUDA, CPU])


class TestSessionInjection(unittest.TestCase):
    """The provider list must reach the sessions rtmlib builds (fake ORT here)."""

    def _fake_ort(self):
        mod = types.ModuleType("onnxruntime")

        class FakeSession:
            made = []

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                FakeSession.made.append(self)

        def _options():
            # mirrors ort.SessionOptions defaults for the two fields we touch
            return types.SimpleNamespace(enable_mem_pattern=True,
                                         execution_mode="parallel")

        mod.InferenceSession = FakeSession
        mod.SessionOptions = _options
        mod.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
        return mod, FakeSession

    def test_requested_providers_replace_the_ones_rtmlib_asks_for(self):
        mod, FakeSession = self._fake_ort()
        with mock.patch.dict(sys.modules, {"onnxruntime": mod}):
            with PB._forced_session_providers([DML, CPU]):
                # rtmlib asks for CPU itself; the backend's choice must win
                mod.InferenceSession(path_or_bytes="model.onnx",
                                     providers=[CPU])
            self.assertIs(mod.InferenceSession, FakeSession)   # restored
        made = FakeSession.made[-1]
        self.assertEqual(made.kwargs["providers"], [DML, CPU])
        # DirectML needs the memory-pattern optimiser off (ORT DML requirement)
        self.assertFalse(made.kwargs["sess_options"].enable_mem_pattern)
        self.assertEqual(made.kwargs["sess_options"].execution_mode, "sequential")

    def test_the_patch_is_removed_even_when_the_build_raises(self):
        mod, FakeSession = self._fake_ort()
        with mock.patch.dict(sys.modules, {"onnxruntime": mod}):
            with self.assertRaises(RuntimeError):
                with PB._forced_session_providers([CPU]):
                    raise RuntimeError("model build failed")
            self.assertIs(mod.InferenceSession, FakeSession)

    def test_cpu_sessions_leave_memory_pattern_alone(self):
        mod, FakeSession = self._fake_ort()
        with mock.patch.dict(sys.modules, {"onnxruntime": mod}):
            with PB._forced_session_providers([CPU]):
                mod.InferenceSession(path_or_bytes="model.onnx")
        self.assertTrue(FakeSession.made[-1].kwargs["sess_options"].enable_mem_pattern)
        self.assertEqual(FakeSession.made[-1].kwargs["providers"], [CPU])


# --------------------------------------------------- backend degradation

class TestBackendDegradesToCpu(_FakeCase):
    def test_unavailable_directml_runs_on_cpu_without_raising(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=CPU_ONLY):
            be = PB.RTMPoseBackend(provider="directml")
            self.assertEqual(be.info["provider"], CPU)   # before the first frame
            out = be.detect(FRAME, 0)
        self.assertIsNotNone(out)
        info = be.info
        self.assertEqual(info["provider"], CPU)
        self.assertEqual(info["engine"], "rtmpose")
        self.assertEqual(info["model"], "BodyWithFeet")
        self.assertEqual(info["provider_requested"], "directml")
        self.assertIn(DML, info["provider_note"])

    def test_unavailable_cuda_runs_on_cpu_without_raising(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=CPU_ONLY):
            be = PB.RTMPoseBackend(provider="cuda")
            self.assertIsNotNone(be.detect(FRAME, 0))
        self.assertEqual(be.info["provider"], CPU)
        self.assertTrue(be._choice.degraded)

    def test_auto_on_a_gpu_machine_requests_directml(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=ALL):
            be = PB.RTMPoseBackend(provider="auto")
            self.assertEqual(be.info["provider"], DML)
            be.detect(FRAME, 0)
        # the fake model has no sessions to inspect, so the resolved choice stands
        self.assertEqual(be.info["provider"], DML)

    def test_unknown_provider_value_still_builds(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=CPU_ONLY):
            be = PB.RTMPoseBackend(provider="quantum")
            self.assertIsNotNone(be.detect(FRAME, 0))
        self.assertEqual(be.info["provider"], CPU)
        self.assertIn("unknown", be.info["provider_note"].lower())

    def test_provider_kwarg_passes_through_make_backend(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=CPU_ONLY):
            be = PB.make_backend("rtmpose", provider="directml")
            self.assertEqual(be.info["provider"], CPU)
        self.assertIsInstance(be, PB.RTMPoseBackend)
        self.assertEqual(be.provider_request, "directml")


# ------------------------------------------------------------ engine info

class TestInfoContract(_FakeCase):
    def test_rtmpose_reports_engine_provider_and_model(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=(DML, CPU)):
            info = PB.make_backend("rtmpose", provider="auto").info
        self.assertEqual(info["engine"], "rtmpose")
        self.assertEqual(info["provider"], DML)
        self.assertEqual(info["model"], "BodyWithFeet")
        self.assertEqual(info["provider_requested"], "auto")

    def test_a_fake_backend_advertises_its_info_dict(self):
        class FakeBackend:
            name = "fake"
            info = {"engine": "fake", "provider": "CPUExecutionProvider",
                    "model": "FakeNet"}

        got = PB.backend_info(FakeBackend())
        self.assertEqual(got, FakeBackend.info)
        got["engine"] = "changed"                 # returned dict is a copy
        self.assertEqual(FakeBackend.info["engine"], "fake")

    def test_backend_without_info_still_yields_a_contract_dict(self):
        self.assertEqual(PB.backend_info(object()),
                         {"engine": "object", "provider": None, "model": None})

    def test_last_built_backend_is_remembered(self):
        with mock.patch.object(PB, "ort_available_providers", return_value=CPU_ONLY):
            PB.make_backend("rtmpose", provider="cpu")
        self.assertEqual(PB.backend_info()["engine"], "rtmpose")
        self.assertEqual(PB.backend_info()["provider"], CPU)


# ------------------------------------------------------- mediapipe default

@unittest.skipUnless(config.MODEL_PATH.exists(), "MediaPipe model not present")
class TestMediaPipeStaysDefault(unittest.TestCase):
    def _make(self, **kwargs):
        from pose_engine import PoseEngine
        engine = PB.make_backend(**kwargs)
        self.assertIsInstance(engine, PoseEngine)
        self.addCleanup(engine.close)
        return engine

    def test_make_backend_mediapipe_returns_the_mediapipe_engine(self):
        engine = self._make(which="mediapipe", model_path=str(config.MODEL_PATH))
        self.assertEqual(engine.info["engine"], "mediapipe")
        self.assertEqual(engine.info["provider"], "mediapipe")
        self.assertEqual(engine.info["model"], "pose_landmarker_full")

    def test_default_backend_is_still_mediapipe(self):
        with mock.patch.object(config, "POSE_BACKEND", "mediapipe"):
            engine = self._make(model_path=str(config.MODEL_PATH))
        self.assertEqual(engine.info["engine"], "mediapipe")

    def test_mediapipe_accepts_and_ignores_the_provider_kwarg(self):
        engine = self._make(which="mediapipe", provider="directml",
                            model_path=str(config.MODEL_PATH))
        self.assertEqual(engine.info["provider"], "mediapipe")

    def test_mediapipe_still_detects_a_frame(self):
        engine = self._make(which="mediapipe", model_path=str(config.MODEL_PATH))
        out = engine.detect(np.zeros((72, 96, 3), np.uint8), 0)
        self.assertTrue(out is None or np.asarray(out).shape == (33, 4))


if __name__ == "__main__":
    unittest.main()


class TestStateExposesPose(unittest.TestCase):
    """A silently downgraded backend must be visible in /api/state."""

    def test_state_has_pose_block(self):
        import app as appmod
        appmod.app.config["TESTING"] = True
        d = appmod.app.test_client().get("/api/state").get_json()
        self.assertIn("pose", d)
        pose = d["pose"]
        self.assertIsInstance(pose, dict)
        self.assertIn("engine", pose)
        self.assertIn("provider", pose)
