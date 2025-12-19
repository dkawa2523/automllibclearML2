import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automl_lib.workflow.preprocessing.meta import build_preprocessing_metadata


class TestPreprocessingMeta(unittest.TestCase):
    def test_build_preprocessing_metadata_writes_artifacts(self) -> None:
        df_raw = pd.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": ["x", "y", None],
                "target": [10.0, 11.0, 12.0],
            }
        )
        df_pre = pd.DataFrame(
            {
                "f0": [0.1, 0.2, 0.3],
                "f1": [1.0, 0.0, 1.0],
                "target": [10.0, 11.0, 12.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            meta = build_preprocessing_metadata(
                output_dir=out_dir,
                run_id="20251213-153012-a1b2c3",
                dataset_key="example",
                target_col="target",
                feature_cols=["a", "b"],
                feature_types={"numeric": ["a"], "categorical": ["b"]},
                preproc_name="standard_onehot",
                cfg_preprocessing={"scaling": ["standard"]},
                df_raw=df_raw,
                df_preprocessed=df_pre,
            )

            artifacts = set(meta.get("artifacts") or [])
            self.assertIn(str(out_dir / "schema.json"), artifacts)
            self.assertIn(str(out_dir / "manifest.json"), artifacts)
            self.assertIn(str(out_dir / "preprocessing" / "recipe.json"), artifacts)
            self.assertIn(str(out_dir / "preprocessing" / "summary.md"), artifacts)

            schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["run_id"], "20251213-153012-a1b2c3")
            self.assertEqual(schema["dataset_key"], "example")
            self.assertEqual(schema["target_column"], "target")
