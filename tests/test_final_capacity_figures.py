import io
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops

from scripts.final_capacity_figures import render_capacity_figures


def _aggregate_rows() -> list[dict[str, object]]:
    rows = []
    for capacity in (64, 100, 128, 160, 200):
        for horizon in (2, 3, 4, 5):
            rows.append(
                {
                    "capacity": capacity,
                    "horizon": horizon,
                    "sequence_count": 1,
                    "peak_occupied_slots_q25": 8.0 + horizon,
                    "peak_occupied_slots_median": 10.0 + horizon,
                    "peak_occupied_slots_q75": 12.0 + horizon,
                    "peak_occupied_slots_max": 20.0 + horizon,
                    "state_bytes": 8 + capacity * 610,
                    "causal_prefix_t_mAP": 0.20 - 0.01 * horizon,
                    "causal_prefix_t_REC": 0.30 - 0.01 * horizon,
                    "normalized_id_switch_rate": 0.05 + 0.01 * horizon,
                    "gap_recovery_recall": (
                        None if horizon == 2 else 0.40 + 0.01 * horizon
                    ),
                }
            )
    return rows


def test_capacity_figures_render_editable_and_nonblank_outputs() -> None:
    payloads = render_capacity_figures(_aggregate_rows(), expected_sequence_count=1)

    assert len(payloads) == 9
    for filename, payload in payloads.items():
        if filename.endswith(".svg"):
            assert ET.fromstring(payload).tag.endswith("svg")
        elif filename.endswith(".pdf"):
            assert payload.startswith(b"%PDF-")
        else:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            assert image.width >= 1000
            assert image.height >= 600
            difference = ImageChops.difference(
                image, Image.new("RGB", image.size, "white")
            )
            assert difference.getbbox() is not None
