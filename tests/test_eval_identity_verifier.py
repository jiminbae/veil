import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from tools.eval_identity_verifier import VerifierError, build_verifier, evaluate_identity


def make_frame(path: Path, target_color, non_target_color) -> None:
    image = Image.new("RGB", (80, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 25, 25), fill=target_color)
    draw.rectangle((45, 5, 70, 30), fill=non_target_color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


class EvalIdentityVerifierTest(unittest.TestCase):
    def test_debug_backend_writes_pair_and_aggregate_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original_frames"
            processed = root / "processed_frames"
            make_frame(original / "frame_000001.png", "red", "blue")
            make_frame(processed / "frame_000001.png", "red", "green")
            reference = root / "target_ref.png"
            Image.new("RGB", (20, 20), "red").save(reference)
            metadata = root / "face_metadata.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "frame_idx": 1,
                            "stable_face_id": 1,
                            "bbox": [5, 5, 25, 25],
                            "is_target_final": True,
                            "is_background": False,
                            "final_action": "PRESERVE",
                        },
                        {
                            "frame_idx": 1,
                            "stable_face_id": 2,
                            "bbox": [45, 5, 70, 30],
                            "is_target_final": False,
                            "is_background": True,
                            "final_action": "BLUR",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            summary = evaluate_identity(
                original=original,
                processed=processed,
                action_metadata=metadata,
                protected_reference=reference,
                output_dir=root / "out",
                backend="debug_color",
                condition="full",
                clip_id="clip_a",
                threshold=0.5,
            )

            self.assertEqual(summary["target_pairs"], 1)
            self.assertEqual(summary["non_target_pairs"], 1)
            self.assertTrue((root / "out" / "pair_scores.csv").exists())
            self.assertTrue((root / "out" / "per_clip_identity_verifier.csv").exists())
            self.assertTrue((root / "out" / "aggregate_identity_verifier.csv").exists())
            config = json.loads((root / "out" / "evaluation_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["paper_valid_backend"])

    def test_adaface_requires_a_real_checkpoint(self):
        with self.assertRaisesRegex(VerifierError, "official local checkpoint"):
            build_verifier("adaface")


if __name__ == "__main__":
    unittest.main()
