import json
from pathlib import Path

import numpy as np
import pytest

from indic_codec_probe.pilot import PilotError
from indic_codec_probe.textgrids import Interval
from indic_codec_probe.unit_features import FrameSeries, build_unit_bundle, overlap_weighted_pool


def _textgrid(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.16
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.16
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.08
            text = "क"
        intervals [2]:
            xmin = 0.08
            xmax = 0.16
            text = "ा"
""",
        encoding="utf-8",
    )


def test_overlap_weighted_pool_uses_actual_overlap() -> None:
    series = FrameSeries(
        values=np.asarray([[0.0], [10.0]], dtype=np.float32),
        starts=np.asarray([0.0, 0.08]),
        ends=np.asarray([0.08, 0.16]),
    )
    pooled = overlap_weighted_pool(series, Interval(0.04, 0.12, "x"))
    assert pooled is not None
    assert pooled.tolist() == pytest.approx([5.0])


def test_invalid_frame_series_is_rejected() -> None:
    series = FrameSeries(
        values=np.asarray([[np.nan]], dtype=np.float32),
        starts=np.asarray([0.0]),
        ends=np.asarray([0.08]),
    )
    with pytest.raises(PilotError, match="non-finite"):
        overlap_weighted_pool(series, Interval(0.0, 0.08, "x"))


def test_build_unit_bundle_keeps_a_common_unit_set(tmp_path: Path) -> None:
    manifest = {
        "language": "Hindi",
        "policy": "codepoint",
        "utterances": [
            {
                "utterance_id": "u1",
                "speaker_id": "s1",
                "split": "train",
                "textgrid_path": "s1/u1.TextGrid",
                "units": ["क", "ा"],
            }
        ],
    }
    manifest_path = tmp_path / "corpus.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _textgrid(tmp_path / "textgrids" / "s1" / "u1.TextGrid")
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    starts = np.asarray([0.0, 0.08])
    ends = np.asarray([0.08, 0.16])
    np.savez_compressed(
        frame_root / "u1.npz",
        representation_names=np.asarray(json.dumps(["a", "b"])),
        a__values=np.asarray([[1.0], [2.0]], dtype=np.float32),
        a__starts=starts,
        a__ends=ends,
        b__values=np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        b__starts=starts,
        b__ends=ends,
    )
    output = tmp_path / "bundle.npz"
    result = build_unit_bundle(manifest_path, tmp_path / "textgrids", frame_root, output)
    assert result["examples"] == 2
    with np.load(output, allow_pickle=False) as bundle:
        assert bundle["a"].shape == (2, 1)
        assert bundle["b"].shape == (2, 2)
