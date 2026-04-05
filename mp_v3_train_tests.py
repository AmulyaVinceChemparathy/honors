#!/usr/bin/env python3
"""
Tests for mp_v3_train.py

Run from repo root:
  python mp_v3_train_tests.py

Tier 1 (always): syntax-check mp_v3_train.py without importing it.
Tier 2 (optional): imports mp_v3_train and tests helpers + model shape if
  tensorflow, opencv, mediapipe, sklearn are installed.
"""
from __future__ import annotations

import os

# Reduce TensorFlow / MediaPipe logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import py_compile
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MP_V3_TRAIN = REPO_ROOT / "mp_v3_train.py"

_DEPS_HINT = (
    'pip install tensorflow opencv-python "mediapipe>=0.10.9,<0.10.31" scikit-learn numpy'
)


# 🔍 Check if local mediapipe.py is shadowing real package
def _mediapipe_shadow_path() -> str | None:
    p = REPO_ROOT / "mediapipe.py"
    return str(p) if p.is_file() else None


# 🔍 Improve import error messages
def _enrich_import_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, AttributeError) and "solutions" in str(exc):
        msg += (
            "\nFix: MediaPipe version incompatible.\n"
            "Run:\n"
            '  pip uninstall -y mediapipe\n'
            '  pip install "mediapipe>=0.10.9,<0.10.31"\n'
        )

        if sh := _mediapipe_shadow_path():
            msg += f"\nAlso remove/rename local file: {sh}"

    return msg


# 🔍 Try importing main file
def _try_import_mp_v3_train():
    try:
        import mp_v3_train
        return mp_v3_train, None
    except Exception as e:
        return None, _enrich_import_error(e)


MP, _MP_IMPORT_ERROR = _try_import_mp_v3_train()
HAS_MP = MP is not None

_SKIP_HELPERS_REASON = _MP_IMPORT_ERROR or "mp_v3_train import failed"

# Show helpful message if import fails
if not HAS_MP and __name__ == "__main__":
    print(f"[INFO] Optional tests skipped:")
    print(_MP_IMPORT_ERROR)
    print(f"[INFO] Install dependencies:\n{_DEPS_HINT}")

    if sh := _mediapipe_shadow_path():
        print(f"[WARNING] Remove shadow file: {sh}")

# Import numpy only if main module loads
if HAS_MP:
    import numpy as np
else:
    np = None


# ==========================================
# ✅ TEST 1: Syntax Check (Always runs)
# ==========================================
class TestMpV3TrainSyntax(unittest.TestCase):
    def test_mp_v3_train_compiles(self):
        py_compile.compile(str(MP_V3_TRAIN), doraise=True)


# ==========================================
# ✅ TEST 2: Functional Tests (Optional)
# ==========================================
@unittest.skipUnless(HAS_MP, _SKIP_HELPERS_REASON)
class TestMpV3TrainHelpers(unittest.TestCase):

    def test_get_all_sign_folders_finds_leaf_with_video(self):
        with tempfile.TemporaryDirectory() as root:
            leaf = os.path.join(root, "Adjectives", "hello")
            os.makedirs(leaf)

            # create dummy video file
            Path(leaf, "clip.mp4").touch()

            folders = MP.get_all_sign_folders(root)

            self.assertIn(
                os.path.abspath(leaf),
                [os.path.abspath(p) for p in folders]
            )

    def test_load_full_npy_dataset_builds_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "class_a")
            b = os.path.join(root, "class_b")

            os.makedirs(a)
            os.makedirs(b)

            dummy = np.zeros((MP.MAX_FRAMES, MP.COMBINED_DIM), dtype=np.float32)

            np.save(os.path.join(a, "v1.npy"), dummy)
            np.save(os.path.join(b, "v2.npy"), dummy)

            X, y, mapping = MP.load_full_npy_dataset(root)

            self.assertEqual(X.shape[0], 2)
            self.assertEqual(len(mapping), 2)
            self.assertEqual(set(y.tolist()), {0, 1})

    def test_build_robust_isl_model_output_shape(self):
        num_classes = 7

        model = MP.build_robust_isl_model(
            MP.MAX_FRAMES,
            MP.COMBINED_DIM,
            num_classes
        )

        self.assertEqual(model.output_shape[-1], num_classes)


# ==========================================
# 🚀 RUN TESTS
# ==========================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
