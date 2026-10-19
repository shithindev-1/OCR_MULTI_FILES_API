import asyncio
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw

from app.core.config import Settings
from app.services.detection import (
    PageCrop,
    _build_detection,
    _gradient_magnitude,
    _overlap_area,
    _polygon_area,
    crop_page,
    edge_support,
    enhance_crop,
    orientation_from_result,
    page_detections,
    refine_box_with_edges,
    refine_detection,
    render_pages,
    rotate_upright,
    run_detection,
    save_crop_outputs,
)


def _corners(x_min: float, y_min: float, x_max: float, y_max: float) -> list[dict[str, float]]:
    return [
        {"x": x_min, "y": y_min},
        {"x": x_max, "y": y_min},
        {"x": x_max, "y": y_max},
        {"x": x_min, "y": y_max},
    ]


def _white_rect_on_black(width: int = 400, height: int = 300, margin: int = 40) -> tuple[Image.Image, bytes]:
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    ImageDraw.Draw(image).rectangle([margin, margin, width - margin, height - margin], fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def _two_cards_side_by_side(width: int = 600, height: int = 300) -> tuple[Image.Image, bytes]:
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 60, 280, 240], fill=(250, 250, 250))
    draw.rectangle([320, 60, 570, 240], fill=(230, 230, 230))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def _single_page_pdf(image: Image.Image) -> bytes:
    width, height = image.size
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(width, height)
    img_obj = pdfium.PdfImage.new(pdf)
    img_obj.set_bitmap(pdfium.PdfBitmap.from_pil(image))
    img_obj.set_matrix(pdfium.PdfMatrix().scale(width, height))
    page.insert_obj(img_obj)
    page.gen_content()
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


class _FakeBoxDetector:
    def __init__(self, pages: list[dict], orientations: list[dict] | None = None) -> None:
        self.pages = pages
        self.orientations = orientations
        self.received_page_counts: list[int] = []
        self.received_crop_counts: list[int] = []

    async def detect_box_corners(self, page_images_png: list[bytes]) -> list[dict]:
        self.received_page_counts.append(len(page_images_png))
        return self.pages

    async def detect_orientations(self, crop_images_png: list[bytes]) -> list[dict]:
        self.received_crop_counts.append(len(crop_images_png))
        if self.orientations is not None:
            return self.orientations
        return [
            {"top_edge": "top", "tilt_degrees": 0, "confidence": 1.0, "reasoning": "upright"}
            for _ in crop_images_png
        ]


class _FailingBoxDetector:
    async def detect_box_corners(self, page_images_png: list[bytes]) -> list[dict]:
        raise RuntimeError("boom")

    async def detect_orientations(self, crop_images_png: list[bytes]) -> list[dict]:
        raise RuntimeError("boom")


class _FailingOrientationDetector(_FakeBoxDetector):
    async def detect_orientations(self, crop_images_png: list[bytes]) -> list[dict]:
        raise RuntimeError("orientation unavailable")


def test_render_pages_rasterizes_pdf_pages() -> None:
    image, _ = _white_rect_on_black()
    content = _single_page_pdf(image)

    pages = render_pages(content, "application/pdf", pdf_dpi=150)

    assert len(pages) == 1
    height, width = pages[0].shape[:2]
    assert height > 0 and width > 0


def test_page_detections_converts_normalized_fractions_to_pixels() -> None:
    detections = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "national id front", "corners": _corners(0.1, 0.1, 0.9, 0.9)}]},
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.method == "gemini_vision"
    assert detection.label == "national id front"
    assert detection.region_number == 1
    assert detection.box == [[40, 30], [360, 30], [360, 270], [40, 270]]
    assert detection.axis_aligned_bbox == [40, 30, 360, 270]


def test_page_detections_returns_one_detection_per_document_in_the_same_frame() -> None:
    detections = page_detections(
        page_number=1,
        width=600,
        height=300,
        page_result={
            "found": True,
            "documents": [
                {"label": "national id front", "corners": _corners(0.05, 0.2, 0.47, 0.8)},
                {"label": "national id back", "corners": _corners(0.53, 0.2, 0.95, 0.8)},
            ],
        },
    )

    assert len(detections) == 2
    assert [detection.region_number for detection in detections] == [1, 2]
    assert [detection.label for detection in detections] == ["national id front", "national id back"]
    assert [detection.method for detection in detections] == ["gemini_vision", "gemini_vision"]
    # The two boxes must stay distinct rather than being merged into one frame-wide box.
    assert detections[0].axis_aligned_bbox == [30, 60, 282, 240]
    assert detections[1].axis_aligned_bbox == [318, 60, 570, 240]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"top_edge": "top", "tilt_degrees": 0, "confidence": 0.9, "reasoning": "upright"},
         (0, 0.9, "upright", "top")),
        # The whole point of the mapping: a top lying along the left edge needs one quarter turn
        # clockwise, one along the right edge needs three. The model no longer does this sum.
        ({"top_edge": "left", "tilt_degrees": 0, "confidence": 0.8, "reasoning": "header at left"},
         (90, 0.8, "header at left", "left")),
        ({"top_edge": "bottom", "tilt_degrees": 0, "confidence": 0.5, "reasoning": "inverted"},
         (180, 0.5, "inverted", "bottom")),
        ({"top_edge": "right", "tilt_degrees": 0, "confidence": 1.0, "reasoning": "header at right"},
         (270, 1.0, "header at right", "right")),
        # Tilt is added on top, so a document that is upside down *and* slanted is corrected in
        # one pass rather than being snapped to the nearest quarter turn.
        ({"top_edge": "bottom", "tilt_degrees": 15, "confidence": 0.8, "reasoning": "inverted + tilt"},
         (195, 0.8, "inverted + tilt", "bottom")),
        ({"top_edge": "top", "tilt_degrees": -7.5, "confidence": 0.7, "reasoning": "slight tilt"},
         (352.5, 0.7, "slight tilt", "top")),
        # Under the deadband: the model reports a degree or two of tilt on nearly every crop, and
        # honouring it would drop every quarter turn off-grid onto the resampling path.
        ({"top_edge": "right", "tilt_degrees": 3, "confidence": 0.97, "reasoning": "noise tilt"},
         (270, 0.97, "noise tilt", "right")),
        ({"top_edge": "bottom", "tilt_degrees": -2, "confidence": 0.98, "reasoning": "noise tilt"},
         (180, 0.98, "noise tilt", "bottom")),
        # A quarter turn in tilt_degrees would double-count what top_edge already carries.
        ({"top_edge": "left", "tilt_degrees": 90, "confidence": 0.9, "reasoning": "doubled up"},
         (90, 0.9, "doubled up", "left")),
        ({"top_edge": "LEFT", "tilt_degrees": 0, "confidence": 0.9, "reasoning": "case insensitive"},
         (90, 0.9, "case insensitive", "left")),
        # Unparseable or missing values must not produce a guessed turn.
        ({"top_edge": "left", "tilt_degrees": "a lot", "confidence": "high", "reasoning": None},
         (90, None, "", "left")),
        ({"top_edge": "sideways", "tilt_degrees": 0, "confidence": 0.9, "reasoning": "unusable"},
         (0, None, "", "sideways")),
        ({}, (0, None, "", "")),
        (None, (0, None, "", "")),
    ],
)
def test_orientation_from_result_normalises_the_models_answer(result, expected) -> None:
    assert orientation_from_result(result) == expected


def test_refine_box_with_edges_snaps_a_bad_quad_onto_the_real_edges() -> None:
    # A card with known, exact bounds.
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    image[150:450, 100:500] = 240

    # A deliberately sloppy quad: roughly over the card but rotated and misplaced, the way the
    # model's estimates actually fail.
    sloppy = [[70, 210], [430, 120], [530, 400], [150, 480]]
    refined = refine_box_with_edges(image, sloppy)

    assert refined is not None
    xs = [point[0] for point in refined]
    ys = [point[1] for point in refined]
    assert abs(min(xs) - 100) <= 6 and abs(max(xs) - 500) <= 6
    assert abs(min(ys) - 150) <= 6 and abs(max(ys) - 450) <= 6


def test_refine_box_with_edges_returns_none_when_nothing_convincing_is_found() -> None:
    # Featureless frame: no contour can be trusted, so the caller keeps the original box.
    image = np.full((400, 400, 3), 128, dtype=np.uint8)

    assert refine_box_with_edges(image, [[50, 50], [350, 50], [350, 350], [50, 350]]) is None


def test_edge_support_scores_a_box_on_the_border_above_one_cutting_through_the_middle() -> None:
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image[80:320, 100:500] = 240
    gradient = _gradient_magnitude(image)

    on_border = [[100, 80], [500, 80], [500, 320], [100, 320]]
    clipped = [[100, 80], [300, 80], [300, 320], [100, 320]]  # right side runs through the card

    assert edge_support(gradient, on_border) > edge_support(gradient, clipped)


def test_refine_detection_rejects_a_fit_that_reaches_into_a_neighbouring_document() -> None:
    # Two cards side by side with a narrow gap, the layout that made refinement overreach.
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    image[40:260, 30:420] = 245
    image[40:260, 470:860] = 235

    left = _build_detection(1, 1, "front", 900, 300, [[30, 40], [420, 40], [420, 260], [30, 260]], "gemini_vision")
    right_box = [[470, 40], [860, 40], [860, 260], [470, 260]]

    refined = refine_detection(image, left, [right_box])

    # Whatever the contour found, it must not end up straddling the neighbour.
    assert _overlap_area(refined.box, right_box) <= _polygon_area(
        [[float(x), float(y)] for x, y in refined.box]
    ) * 0.02


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    # Pixels stored one way, EXIF saying to display them rotated 180° — what phone cameras do.
    image = Image.new("RGB", (120, 60), (10, 10, 10))
    ImageDraw.Draw(image).rectangle([0, 0, 119, 14], fill=(250, 250, 250))  # bright band on top
    exif = image.getexif()
    exif[274] = 3
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)

    loaded = render_pages(path.read_bytes(), "image/jpeg", 200)[0]

    # After honouring the tag the bright band belongs at the bottom, as a viewer would show it.
    top_band = loaded[:15].mean()
    bottom_band = loaded[-15:].mean()
    assert bottom_band > top_band


def test_refine_detection_keeps_the_original_box_for_fallback_methods() -> None:
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[100:300, 100:300] = 255
    detection = page_detections(page_number=1, width=400, height=400, page_result={"found": False, "documents": []})[0]

    assert refine_detection(image, detection) is detection


def test_page_detections_falls_back_when_not_found() -> None:
    detections = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": False, "documents": [{"label": "full frame", "corners": _corners(0, 0, 1, 1)}]},
    )

    assert len(detections) == 1
    assert detections[0].method == "fallback_full_frame"
    assert detections[0].axis_aligned_bbox == [0, 0, 400, 300]


def test_page_detections_falls_back_on_degenerate_quad() -> None:
    detections = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "unknown", "corners": [{"x": 0.5, "y": 0.5}] * 4}]},
    )

    assert detections[0].method == "fallback_invalid_response"


def test_page_detections_falls_back_when_documents_list_is_empty() -> None:
    detections = page_detections(page_number=1, width=400, height=300, page_result={"found": True, "documents": []})

    assert len(detections) == 1
    assert detections[0].method == "fallback_full_frame"
    assert detections[0].label == "full frame"


def test_crop_page_straightens_to_detected_bounds() -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]},
    )[0]

    cropped = crop_page(images[0], detection)

    height, width = cropped.shape[:2]
    assert 300 <= width <= 335
    assert 200 <= height <= 235


def test_crop_page_padding_widens_the_crop_on_every_side() -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]},
    )[0]

    unpadded = crop_page(images[0], detection)
    padded = crop_page(images[0], detection, padding=10)

    assert padded.shape[0] == unpadded.shape[0] + 20
    assert padded.shape[1] == unpadded.shape[1] + 20


def test_crop_page_padding_ratio_scales_with_the_document_and_overrides_the_floor() -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]},
    )[0]

    unpadded = crop_page(images[0], detection)
    # The ratio is taken off the document's longest edge (320px here), so 10% = 32px per side,
    # which outweighs the 4px floor and applies equally to the short axis.
    padded = crop_page(images[0], detection, padding=4, padding_ratio=0.1)
    expected_margin = round(max(unpadded.shape[:2]) * 0.1)

    assert expected_margin > 4
    assert padded.shape[1] == unpadded.shape[1] + 2 * expected_margin
    assert padded.shape[0] == unpadded.shape[0] + 2 * expected_margin


def test_crop_page_top_padding_ratio_adds_extra_headroom_above_only() -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]},
    )[0]

    unpadded = crop_page(images[0], detection)
    even = crop_page(images[0], detection, padding_ratio=0.05)
    top_biased = crop_page(images[0], detection, padding_ratio=0.05, top_padding_ratio=0.12)

    longest_edge = max(unpadded.shape[:2])
    side_margin = round(longest_edge * 0.05)
    top_margin = round(longest_edge * 0.12)

    # Width is untouched by the top ratio; only the vertical axis grows, and only above.
    assert top_biased.shape[1] == even.shape[1] == unpadded.shape[1] + 2 * side_margin
    assert top_biased.shape[0] == unpadded.shape[0] + top_margin + side_margin
    assert top_biased.shape[0] > even.shape[0]


def test_crop_page_top_margin_never_falls_below_the_other_sides() -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(
        page_number=1,
        width=400,
        height=300,
        page_result={"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]},
    )[0]

    unpadded = crop_page(images[0], detection)
    # A top ratio smaller than the base ratio must not shrink the top margin below the sides.
    crop = crop_page(images[0], detection, padding_ratio=0.05, top_padding_ratio=0.01)

    side_margin = round(max(unpadded.shape[:2]) * 0.05)
    assert crop.shape[0] == unpadded.shape[0] + 2 * side_margin


def test_save_crop_outputs_sanitizes_path_traversal_in_filename(tmp_path: Path) -> None:
    _, content = _white_rect_on_black()
    images = render_pages(content, "image/png", pdf_dpi=200)
    detection = page_detections(page_number=1, width=400, height=300, page_result={"found": False, "documents": []})[0]
    crop = PageCrop(detection=detection, cropped_image=crop_page(images[0], detection))

    saved_paths = save_crop_outputs(
        output_dir=tmp_path,
        filename="../../etc/passwd.png",
        sha256="abcdef1234567890",
        document_type="national_id",
        crops=[crop],
    )

    assert saved_paths[0].parent == tmp_path
    assert ".." not in saved_paths[0].name


def test_run_detection_saves_crop_and_metadata_using_gemini_response(tmp_path: Path) -> None:
    _, content = _white_rect_on_black()
    settings = Settings(gemini_api_key="test", gemini_model="test-model", detection_output_dir=str(tmp_path))
    fake_detector = _FakeBoxDetector(
        [{"found": True, "documents": [{"label": "national id front", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]}]
    )

    saved_paths = asyncio.run(
        run_detection(
            content=content,
            filename="national-id-front.png",
            mime_type="image/png",
            document_type="national_id",
            sha256="abcdef1234567890",
            settings=settings,
            gemini_service=fake_detector,
        )
    )

    assert saved_paths is not None
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert fake_detector.received_page_counts == [1]

    metadata = json.loads((tmp_path / "national-id-front_abcdef12.json").read_text(encoding="utf-8"))
    page = metadata["pages"][0]
    assert page["documents_found"] == 1
    # Edge refinement runs on top of the model's box, so either geometry source is acceptable.
    assert page["documents"][0]["method"] in ("gemini_vision", "gemini_vision+edges")
    assert page["documents"][0]["label"] == "national id front"
    assert page["documents"][0]["cropped_file"] == saved_paths[0].name


def test_run_detection_saves_one_crop_per_document_found_in_one_image(tmp_path: Path) -> None:
    _, content = _two_cards_side_by_side()
    # Enhancement off, so this test pins crop geometry (padding plumbing) in isolation.
    settings = Settings(
        gemini_api_key="test",
        gemini_model="test-model",
        detection_output_dir=str(tmp_path),
        detection_crop_target_long_edge=0,
        detection_crop_sharpen_amount=0.0,
        detection_crop_contrast_clip=0.0,
    )
    fake_detector = _FakeBoxDetector(
        [
            {
                "found": True,
                "documents": [
                    {"label": "national id front", "corners": _corners(0.05, 0.2, 0.467, 0.8)},
                    {"label": "national id back", "corners": _corners(0.533, 0.2, 0.95, 0.8)},
                ],
            }
        ]
    )

    saved_paths = asyncio.run(
        run_detection(
            content=content,
            filename="national-id.png",
            mime_type="image/png",
            document_type="national_id",
            sha256="abcdef1234567890",
            settings=settings,
            gemini_service=fake_detector,
        )
    )

    assert saved_paths is not None
    assert [path.name for path in saved_paths] == [
        "national-id_abcdef12_page1_doc1.png",
        "national-id_abcdef12_page1_doc2.png",
    ]
    # Each crop covers only its own 250px-wide card, never the full 600px frame, and carries the
    # configured margin on both sides (so this also proves the settings reach crop_page).
    # Each crop covers only its own ~250x180 card plus the configured margin, never the full
    # 600px frame. Edge refinement can shift the box by a few pixels, so allow a small tolerance.
    expected_margin = round(max(float(settings.detection_crop_padding_px), 250 * settings.detection_crop_padding_ratio))
    for path in saved_paths:
        with Image.open(path) as cropped:
            width, height = cropped.size
            assert abs(width - (250 + 2 * expected_margin)) <= 8
            assert abs(height - (180 + 2 * expected_margin)) <= 8

    metadata = json.loads((tmp_path / "national-id_abcdef12.json").read_text(encoding="utf-8"))
    page = metadata["pages"][0]
    assert page["documents_found"] == 2
    assert [document["label"] for document in page["documents"]] == ["national id front", "national id back"]
    assert [document["region_number"] for document in page["documents"]] == [1, 2]
    assert [document["cropped_file"] for document in page["documents"]] == [path.name for path in saved_paths]


@pytest.mark.parametrize(
    ("rotation", "expected_shape"),
    [(0, (60, 200)), (90, (200, 60)), (180, (60, 200)), (270, (200, 60)), (360, (60, 200))],
)
def test_rotate_upright_uses_lossless_quarter_turns(rotation, expected_shape) -> None:
    image = np.zeros((60, 200, 3), dtype=np.uint8)

    rotated = rotate_upright(image, rotation)

    assert rotated.shape[:2] == expected_shape


@pytest.mark.parametrize("rotation", [None, "ninety"])
def test_rotate_upright_leaves_the_image_alone_for_unusable_angles(rotation) -> None:
    image = np.zeros((60, 200, 3), dtype=np.uint8)

    assert rotate_upright(image, rotation).shape[:2] == (60, 200)


@pytest.mark.parametrize("rotation", [15, 195, 352.5])
def test_rotate_upright_handles_off_grid_angles_without_clipping_corners(rotation) -> None:
    image = np.full((60, 200, 3), 255, dtype=np.uint8)

    rotated = rotate_upright(image, rotation)

    # An arbitrary rotation must grow the canvas to the rotated bounding box, never crop to fit.
    height, width = rotated.shape[:2]
    assert height > 60 and width > 200


def test_rotate_upright_off_grid_angle_actually_tilts_content() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[50:70, :] = 255  # horizontal white band

    rotated = rotate_upright(image, 20)

    # After a 20° turn the band is no longer confined to the rows it occupied before.
    band_rows = np.where((rotated > 200).any(axis=(1, 2)))[0]
    assert band_rows.max() - band_rows.min() > 40


def test_rotate_upright_actually_reorients_content() -> None:
    image = np.zeros((40, 100, 3), dtype=np.uint8)
    image[0, :] = 255  # bright top row

    rotated = rotate_upright(image, 90)

    # A clockwise quarter turn sends the top row to the right-hand column.
    assert rotated.shape[:2] == (100, 40)
    assert rotated[:, -1].mean() > 200


def test_enhance_crop_scales_toward_the_target_long_edge() -> None:
    small = np.full((100, 200, 3), 128, dtype=np.uint8)

    enlarged = enhance_crop(small, 0, target_long_edge=600)

    assert max(enlarged.shape[:2]) == 600
    assert enlarged.shape[0] / enlarged.shape[1] == pytest.approx(100 / 200, rel=0.02)


def test_enhance_crop_downscales_oversized_crops() -> None:
    large = np.full((3000, 4000, 3), 128, dtype=np.uint8)

    reduced = enhance_crop(large, 0, target_long_edge=1600)

    assert max(reduced.shape[:2]) == 1600


def test_enhance_crop_caps_upscaling_of_tiny_crops() -> None:
    tiny = np.full((20, 30, 3), 128, dtype=np.uint8)

    enlarged = enhance_crop(tiny, 0, target_long_edge=1600)

    # 1600/30 would be ~53x; capped so a tiny crop is made legible, not fabricated.
    assert max(enlarged.shape[:2]) == 90


def test_enhance_crop_rotates_before_sizing_so_the_target_applies_to_final_orientation() -> None:
    image = np.full((100, 400, 3), 128, dtype=np.uint8)

    enhanced = enhance_crop(image, 90, target_long_edge=800)

    assert enhanced.shape[0] > enhanced.shape[1]
    assert max(enhanced.shape[:2]) == 800


def test_enhance_crop_sharpening_increases_edge_contrast() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 200
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)

    soft = enhance_crop(blurred, 0, sharpen_amount=0.0)
    sharp = enhance_crop(blurred, 0, sharpen_amount=1.2)

    def edge_energy(array: np.ndarray) -> float:
        return float(cv2.Laplacian(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    assert edge_energy(sharp) > edge_energy(soft)


def test_run_detection_rotates_using_the_post_crop_orientation_pass(tmp_path: Path) -> None:
    _, content = _white_rect_on_black(width=400, height=300)
    settings = Settings(
        gemini_api_key="test",
        gemini_model="test-model",
        detection_output_dir=str(tmp_path),
        detection_crop_target_long_edge=900,
    )
    fake_detector = _FakeBoxDetector(
        [{"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]}],
        orientations=[
            {"top_edge": "left", "tilt_degrees": 0, "confidence": 0.93, "reasoning": "header at left"}
        ],
    )

    saved_paths = asyncio.run(
        run_detection(
            content=content,
            filename="rotated.png",
            mime_type="image/png",
            document_type="national_id",
            sha256="abcdef1234567890",
            settings=settings,
            gemini_service=fake_detector,
        )
    )

    assert saved_paths is not None
    # Orientation is judged on the cropped document, so exactly one crop was sent for review.
    assert fake_detector.received_crop_counts == [1]
    with Image.open(saved_paths[0]) as saved:
        width, height = saved.size
    # The source card is landscape; a 90° turn makes the saved file portrait, sized to target.
    assert height > width
    assert max(width, height) == 900

    document = json.loads((tmp_path / "rotated_abcdef12.json").read_text(encoding="utf-8"))["pages"][0]["documents"][0]
    assert document["rotation_degrees"] == 90
    assert document["rotation_confidence"] == 0.93
    assert document["rotation_reason"] == "header at left"


def test_run_detection_still_saves_crops_when_the_orientation_pass_fails(tmp_path: Path) -> None:
    _, content = _white_rect_on_black(width=400, height=300)
    settings = Settings(gemini_api_key="test", gemini_model="test-model", detection_output_dir=str(tmp_path))
    detector = _FailingOrientationDetector(
        [{"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]}]
    )

    saved_paths = asyncio.run(
        run_detection(
            content=content,
            filename="unrotated.png",
            mime_type="image/png",
            document_type="national_id",
            sha256="abcdef1234567890",
            settings=settings,
            gemini_service=detector,
        )
    )

    # A failed orientation pass must not cost us the crop; it just stays as-is.
    assert saved_paths is not None and saved_paths[0].exists()
    document = json.loads((tmp_path / "unrotated_abcdef12.json").read_text(encoding="utf-8"))["pages"][0]["documents"][0]
    assert document["rotation_degrees"] == 0
    assert document["rotation_confidence"] is None


def test_run_detection_logs_each_rotation_as_its_own_json_record(tmp_path: Path, caplog) -> None:
    _, content = _white_rect_on_black(width=400, height=300)
    settings = Settings(gemini_api_key="test", gemini_model="test-model", detection_output_dir=str(tmp_path))
    detector = _FakeBoxDetector(
        [{"found": True, "documents": [{"label": "card", "corners": _corners(0.1, 0.1333, 0.9, 0.8667)}]}],
        orientations=[
            {"top_edge": "bottom", "tilt_degrees": 15, "confidence": 0.82, "reasoning": "inverted + 15 tilt"}
        ],
    )

    with caplog.at_level("INFO", logger="uae_ocr.rotation"):
        asyncio.run(
            run_detection(
                content=content,
                filename="tilted.png",
                mime_type="image/png",
                document_type="aadhar",
                sha256="abcdef1234567890",
                settings=settings,
                gemini_service=detector,
            )
        )

    records = [json.loads(record.message) for record in caplog.records if record.name == "uae_ocr.rotation"]
    assert len(records) == 1
    entry = records[0]
    assert entry["rotation_requested_degrees"] == 195
    assert entry["rotation_confidence"] == 0.82
    assert entry["rotation_reason"] == "inverted + 15 tilt"
    assert entry["rotation_applied"] is True
    # 195° is not a quarter turn, so it must go through the affine path, not cv2.rotate.
    assert entry["rotation_method"] == "cv2.warpAffine"
    assert entry["document_type"] == "aadhar"
    assert entry["label"] == "card"
    assert entry["size_before"] and entry["size_after"]


def test_run_detection_returns_none_and_writes_nothing_on_detector_failure(tmp_path: Path) -> None:
    _, content = _white_rect_on_black()
    settings = Settings(gemini_api_key="test", gemini_model="test-model", detection_output_dir=str(tmp_path))

    result = asyncio.run(
        run_detection(
            content=content,
            filename="x.png",
            mime_type="image/png",
            document_type="national_id",
            sha256="abcdef1234567890",
            settings=settings,
            gemini_service=_FailingBoxDetector(),
        )
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_split_regions_may_be_far_smaller_than_the_box_calls_minimum() -> None:
    # One ID card on an A4 scan covers under 4% of it.
    small = {"found": True, "documents": [{"label": "card", "corners": _corners(0.649, 0.239, 0.948, 0.365)}]}

    assert page_detections(page_number=1, width=2337, height=3300, page_result=small)[0].method == "fallback_invalid_response"
    assert page_detections(page_number=1, width=2337, height=3300, page_result=small,
                           min_area_ratio=0.005)[0].method == "gemini_vision"
