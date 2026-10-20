"""Document box detection, cropping, rotation and enhancement of uploaded pages."""

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from app.core.config import Settings
from app.core.logging import log_timestamps

logger = logging.getLogger("uae_ocr")
rotation_logger = logging.getLogger("uae_ocr.rotation")

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_MIN_VALID_AREA_RATIO = 0.05
# Largest corner value still read as a fraction: the model is asked to place an uncertain corner
# slightly outside the edge, so 1.002 is a fraction just past the edge, not the 0-1000 scale.
_MAX_FRACTION = 1.5
# Share of BOTH boxes two regions on one page must overlap by to count as the same document.
_DUPLICATE_OVERLAP_RATIO = 0.8
_DETECTION_MAX_DIMENSION = 1024
_MAX_UPSCALE = 3.0
# Pages of a PDF built from crops are stored as JPEG; Pillow's default (quality 75, halved colour
# resolution) visibly blurs small print, so they are saved near-lossless with full colour detail.
PDF_PAGE_QUALITY = {"quality": 95, "subsampling": 0}
_VALID_ROTATIONS = (0, 90, 180, 270)
# Where the document's top currently sits -> the clockwise turn that brings it back to the top.
# A top lying along the left edge needs a quarter turn clockwise; one along the right edge needs
# three. This is the whole of the arithmetic the model used to be asked to do in its head.
TOP_EDGE_ROTATIONS = {"top": 0, "right": 270, "bottom": 180, "left": 90}
_MAX_TILT_DEGREES = 45
# A crop that has already been perspective-warped onto its own corners has no real slant left, yet
# the model reports one or two degrees of it on almost every document. Ignoring anything under
# this keeps the common case on cv2.rotate's lossless quarter turn instead of resampling through
# warpAffine, and keeps repeat runs byte-identical; a genuine slant is far larger than this.
_TILT_DEADBAND_DEGREES = 5
_REFINE_ROI_EXPANSION = 0.12
_REFINE_MIN_ROI_RATIO = 0.15
_REFINE_AREA_BOUNDS = (0.35, 1.6)
_REFINE_CANNY_THRESHOLDS = ((30, 90), (50, 150), (20, 60))
# A refined quad has to trace the border clearly better than the model's, not merely differently.
_REFINE_SUPPORT_MARGIN = 1.05
_REFINE_MAX_SIBLING_OVERLAP = 0.02


class DocumentVision(Protocol):
    async def detect_box_corners(self, page_images_png: list[bytes]) -> list[dict[str, Any]]: ...
    async def detect_orientations(self, crop_images_png: list[bytes]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PageDetection:
    page_number: int
    region_number: int
    label: str
    width: int
    height: int
    box: list[list[int]]
    axis_aligned_bbox: list[int]
    angle_degrees: float
    method: str


@dataclass(frozen=True)
class PageCrop:
    detection: PageDetection
    cropped_image: np.ndarray
    # Orientation is judged after cropping, from the isolated document, so it belongs to the
    # crop rather than to the box detection that produced it.
    rotation_degrees: float = 0.0
    rotation_confidence: float | None = None
    rotation_reason: str = ""
    # The raw observation the angle was derived from, kept for the log so a bad rotation can be
    # traced to a misread edge rather than to the mapping.
    rotation_top_edge: str = ""
    raw_size: tuple[int, int] = (0, 0)


def _load_image(content: bytes) -> np.ndarray:
    with Image.open(BytesIO(content)) as image:
        # Honour the EXIF orientation tag. Phone cameras and scanners routinely store pixels in
        # one orientation and a tag saying how to display them; viewers apply it, PIL does not.
        # Without this the pipeline would work on a differently-oriented image than the one the
        # user is looking at, so reported rotations and left/right labels would not match.
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


# PDFium is not thread-safe, and PDFs are rendered in worker threads - several files of one request
# at once. Two renders at the same time crashed with "access violation" and every file of the request
# failed as unreadable, so all PDFium work runs one document at a time.
_PDFIUM_LOCK = threading.Lock()


def _render_pdf_pages(content: bytes, dpi: int) -> list[np.ndarray]:
    with _PDFIUM_LOCK:
        return _render_pdf_pages_locked(content, dpi)


def _render_pdf_pages_locked(content: bytes, dpi: int) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(content)
    try:
        scale = dpi / 72
        pages = []
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
            pages.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return pages
    finally:
        pdf.close()


def render_pages(content: bytes, mime_type: str, pdf_dpi: int) -> list[np.ndarray]:
    if mime_type == "application/pdf":
        return _render_pdf_pages(content, pdf_dpi)
    return [_load_image(content)]


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode page as PNG")
    return buffer.tobytes()


def downscale_for_detection(image_bgr: np.ndarray, max_dimension: int = _DETECTION_MAX_DIMENSION) -> np.ndarray:
    # The box-detection call only needs enough resolution to localize a rectangle, not full
    # print quality — downscaling here cuts image tokens/latency on that call while the crop
    # itself still uses the full-resolution array (box coordinates are resolution-independent
    # fractions, so this is transparent to corners_to_detection).
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale >= 1.0:
        return image_bgr
    return cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _encode_pages_for_detection(images: list[np.ndarray]) -> list[bytes]:
    return [encode_png(downscale_for_detection(image)) for image in images]


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def _axis_scale(values: list[float]) -> float | None:
    """`_corner_scale` for one axis: the model has answered x as fractions and y on 0-1000 in one reply."""
    largest = max(values)
    if largest <= _MAX_FRACTION:
        return 1.0
    if largest <= 1000.0:
        return 1000.0
    return None


def _card_box(corners: list[Any], width: int, height: int) -> list[list[int]] | None:
    """The box of the one card on a merged National ID page (its front and back sent as two files): the
    rectangle around the answered corners, each axis read on its own scale. The model has answered these
    with x and y on different scales, in a crossed order, or with one corner far off - any of which made
    the outline collapse and the page be taken whole. The rectangle holds the whole card; the edge
    refinement then snaps it onto the card's real, possibly tilted, edges."""
    try:
        xs = [float(corner["x"]) for corner in corners]
        ys = [float(corner["y"]) for corner in corners]
    except (KeyError, TypeError, ValueError):
        return None
    x_scale, y_scale = _axis_scale(xs), _axis_scale(ys)
    if x_scale is None or y_scale is None:
        return None
    left, right = (min(max(round(value / x_scale * width), 0), width) for value in (min(xs), max(xs)))
    top, bottom = (min(max(round(value / y_scale * height), 0), height) for value in (min(ys), max(ys)))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _corner_scale(corners: list[Any]) -> float | None:
    """The scale a corner answer is on: 1 for the fractions the prompt asks for, 1000 when the model
    answers on its own 0-1000 scale instead (it switches between the two from one call to the next).
    Anything larger is a pixel answer on an unknown image size - None, so the full frame is used.
    Read from the largest value: a real document box always reaches well past 1 on a 0-1000 scale,
    while a fraction may run a little past 1 for a corner placed just outside the image edge."""
    values = [float(corner[axis]) for corner in corners for axis in ("x", "y")]
    largest = max(values)
    if largest <= _MAX_FRACTION:
        return 1.0
    if largest <= 1000.0:
        return 1000.0
    return None


def _detection_from_corners(
    *,
    page_number: int,
    region_number: int,
    label: str,
    width: int,
    height: int,
    corners: Any,
    found: bool,
    min_area_ratio: float = _MIN_VALID_AREA_RATIO,
    order_corners: bool = False,
) -> PageDetection:
    box: list[list[int]] | None = None
    if order_corners:
        card = _card_box(corners, width, height) if found and isinstance(corners, list) and len(corners) == 4 else None
        if card is not None and _polygon_area(card) >= width * height * min_area_ratio:
            box = card
    scale = (
        _corner_scale(corners) if not order_corners and found and isinstance(corners, list) and len(corners) == 4
        else None
    )
    if scale is not None:
        pixel_corners = [
            [
                min(max(round(float(corner["x"]) / scale * width), 0), width),
                min(max(round(float(corner["y"]) / scale * height), 0), height),
            ]
            for corner in corners
        ]
        if _polygon_area(pixel_corners) >= width * height * min_area_ratio:
            box = pixel_corners

    if box is None:
        box = [[0, 0], [width, 0], [width, height], [0, height]]
        method = "fallback_full_frame" if not found else "fallback_invalid_response"
    else:
        method = "gemini_vision"

    return _build_detection(page_number, region_number, label, width, height, box, method)


def _build_detection(
    page_number: int,
    region_number: int,
    label: str,
    width: int,
    height: int,
    box: list[list[int]],
    method: str,
) -> PageDetection:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    axis_aligned_bbox = [min(xs), min(ys), max(xs), max(ys)]
    top_left, top_right = box[0], box[1]
    angle_degrees = round(float(np.degrees(np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))), 2)
    return PageDetection(
        page_number, region_number, label, width, height, box, axis_aligned_bbox, angle_degrees, method
    )


def refine_box_with_edges(image_bgr: np.ndarray, box: list[list[int]]) -> list[list[int]] | None:
    """Snap an approximate quad onto the document's real edges.

    Gemini reliably says *which* region holds a document, but its corner coordinates are only
    roughly placed — often tens of degrees off the true edges. A wrong quad makes the perspective
    warp bake in a shear instead of removing one, which shows up as a crop that is still tilted
    and padded with background. Searching for the strongest contour *inside* the reported region
    keeps the model's semantics and replaces only the geometry. Returns None when nothing
    convincing is found, so the caller can keep the original box.
    """
    height, width = image_bgr.shape[:2]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    model_area = _polygon_area([[float(x), float(y)] for x, y in box])
    if model_area <= 0:
        return None

    # Widen the search window so the true edges are inside it even when the quad is badly placed.
    pad_x = (max(xs) - min(xs)) * _REFINE_ROI_EXPANSION
    pad_y = (max(ys) - min(ys)) * _REFINE_ROI_EXPANSION
    x0 = max(int(min(xs) - pad_x), 0)
    y0 = max(int(min(ys) - pad_y), 0)
    x1 = min(int(max(xs) + pad_x), width)
    y1 = min(int(max(ys) + pad_y), height)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi_area = roi.shape[0] * roi.shape[1]
    best_contour = None
    best_area = 0.0
    for low, high in _REFINE_CANNY_THRESHOLDS:
        edges = cv2.dilate(cv2.Canny(blurred, low, high), np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= roi_area * _REFINE_MIN_ROI_RATIO and area > best_area:
                best_contour, best_area = contour, area
    if best_contour is None:
        return None

    points = cv2.boxPoints(cv2.minAreaRect(best_contour))
    refined = [[int(round(x + x0)), int(round(y + y0))] for x, y in points]
    refined_area = _polygon_area([[float(x), float(y)] for x, y in refined])
    low_bound, high_bound = _REFINE_AREA_BOUNDS
    # Reject a fit that bears no relation to the region Gemini pointed at — that means the
    # contour locked onto the background or a fragment, not the document.
    if not model_area * low_bound <= refined_area <= model_area * high_bound:
        return None
    return refined


def _gradient_magnitude(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))


def edge_support(gradient: np.ndarray, box: list[list[int]]) -> float:
    """How strongly a quad's four sides sit on real image edges, scored by its weakest side.

    A box that genuinely traces a document's border runs along strong gradients the whole way
    round. One that clips through the middle of a card has an interior side with almost no
    gradient under it, so taking the minimum across the four sides exposes exactly that failure.
    """
    height, width = gradient.shape
    points = [(float(x), float(y)) for x, y in box]
    scores = []
    for index in range(4):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % 4]
        samples = max(int(round(float(np.hypot(x1 - x0, y1 - y0)))), 2)
        xs = np.clip(np.linspace(x0, x1, samples).astype(int), 0, width - 1)
        ys = np.clip(np.linspace(y0, y1, samples).astype(int), 0, height - 1)
        scores.append(float(gradient[ys, xs].mean()))
    return min(scores)


def _bounds(box: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_area(first: list[list[int]], second: list[list[int]]) -> float:
    ax0, ay0, ax1, ay1 = _bounds(first)
    bx0, by0, bx1, by1 = _bounds(second)
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))


def refine_detection(
    image_bgr: np.ndarray,
    detection: PageDetection,
    sibling_boxes: list[list[list[int]]] = (),
) -> PageDetection:
    if detection.method != "gemini_vision":
        return detection
    refined = refine_box_with_edges(image_bgr, detection.box)
    if refined is None:
        return detection

    # Two documents in one frame never overlap, so a fit that reaches into a neighbour's region
    # has locked onto the wrong thing — common when cards sit side by side with only a thin gap,
    # since the search window necessarily extends past the gap.
    refined_area = _polygon_area([[float(x), float(y)] for x, y in refined])
    if refined_area > 0 and any(
        _overlap_area(refined, sibling) > refined_area * _REFINE_MAX_SIBLING_OVERLAP
        for sibling in sibling_boxes
    ):
        return detection

    # Area alone cannot tell a good fit from one that is merely similar in size but clipped or
    # shifted, so compare how well each candidate's sides lie on actual edges. Documents that
    # touch each other or run to the frame border have no clean outer boundary, and there the
    # contour tends to lock onto an interior feature — which this check rejects.
    gradient = _gradient_magnitude(image_bgr)
    if edge_support(gradient, refined) <= edge_support(gradient, detection.box) * _REFINE_SUPPORT_MARGIN:
        return detection
    return _build_detection(
        detection.page_number,
        detection.region_number,
        detection.label,
        detection.width,
        detection.height,
        refined,
        "gemini_vision+edges",
    )


def _refine_pairs(pairs: list[tuple[PageDetection, np.ndarray]]) -> list[tuple[PageDetection, np.ndarray]]:
    refined = []
    for index, (detection, image) in enumerate(pairs):
        siblings = [
            other.box
            for position, (other, _) in enumerate(pairs)
            if position != index and other.page_number == detection.page_number
        ]
        refined.append((refine_detection(image, detection, siblings), image))
    return refined


def orientation_from_result(result: Any) -> tuple[float, float | None, str, str]:
    """Turn one orientation answer into the clockwise angle to apply.

    The model reports only where the document's top currently lies; the quarter turn that
    corrects it is looked up here rather than asked for. That split exists because the lookup is
    the step the model got wrong — repeat runs on an identical crop agreed on what they saw and
    still returned opposite angles, since a structured answer commits to its first field before
    any reasoning is written. Perception stays with the model, arithmetic stays here, where it is
    exact and testable. An unusable answer means "leave it as it is" rather than a guessed turn.
    """
    if not isinstance(result, dict):
        return 0.0, None, "", ""
    top_edge = str(result.get("top_edge") or "").strip().lower()
    base = TOP_EDGE_ROTATIONS.get(top_edge)
    if base is None:
        return 0.0, None, "", top_edge
    try:
        tilt = float(result.get("tilt_degrees"))
    except (TypeError, ValueError):
        tilt = 0.0
    if not -_MAX_TILT_DEGREES <= tilt <= _MAX_TILT_DEGREES:
        # A "tilt" that large is a quarter turn smuggled back in, which top_edge already carries;
        # applying both would double-count it, so an out-of-range value is dropped instead.
        tilt = 0.0
    elif abs(tilt) < _TILT_DEADBAND_DEGREES:
        tilt = 0.0
    rotation = round((base + tilt) % 360, 2)
    try:
        confidence: float | None = float(result.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    return rotation, confidence, str(result.get("reasoning") or "").strip()[:200], top_edge


def _drop_duplicate_regions(detections: list[PageDetection]) -> list[PageDetection]:
    """Keeps one region where the model reported the same document twice on a page.

    Two boxes count as the same document only when they are nearly the same box - their overlap
    covers most of BOTH of them - because two separate physical documents can never occupy the
    same area of one photo. Containment alone is not enough: a box the model drew too large (the
    whole page, around a card's front and back) contains a real, separate document, and treating
    that as a duplicate would silently drop it. So this only ever removes an exact repeat, and an
    oversized box costs at most an extra crop, never a lost document. Survivors keep their original
    top-to-bottom order and are renumbered from 1.
    """
    if len(detections) < 2:
        return detections

    def area(detection: PageDetection) -> float:
        x0, y0, x1, y1 = detection.axis_aligned_bbox
        return float(max(0, x1 - x0) * max(0, y1 - y0))

    kept: list[PageDetection] = []
    for detection in sorted(detections, key=area, reverse=True):
        duplicate_of = next(
            (
                other for other in kept
                if _overlap_area(detection.box, other.box)
                >= _DUPLICATE_OVERLAP_RATIO * max(area(detection), area(other))
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(detection)
            continue
        logger.warning(
            "Duplicate detection dropped page=%d region=%d label=%s kept_region=%d kept_label=%s",
            detection.page_number, detection.region_number, detection.label,
            duplicate_of.region_number, duplicate_of.label,
        )

    kept.sort(key=lambda detection: detection.region_number)
    return [replace(detection, region_number=number) for number, detection in enumerate(kept, start=1)]


def page_detections(
    *,
    page_number: int,
    width: int,
    height: int,
    page_result: dict[str, Any],
    drop_duplicates: bool = True,
    min_area_ratio: float = _MIN_VALID_AREA_RATIO,
    order_corners: bool = False,
) -> list[PageDetection]:
    """`min_area_ratio`: the smallest share of the page a box may cover before it is taken as a broken
    answer. The split step passes a far smaller value - one ID card on an A4 scan is under 4% of it.
    `order_corners`: a National ID's merged front/back - each page's box is read with `_card_box`."""
    # One page/frame can hold several separate physical documents (a card's front and back
    # photographed side by side, say), so each entry Gemini reports becomes its own detection —
    # and its own crop — rather than being merged into a single box covering both.
    found = bool(page_result.get("found"))
    documents = page_result.get("documents")
    detections: list[PageDetection] = []
    if found and isinstance(documents, list):
        for region_number, document in enumerate(documents, start=1):
            if not isinstance(document, dict):
                continue
            label = str(document.get("label") or "document").strip()[:60] or "document"
            detections.append(
                _detection_from_corners(
                    page_number=page_number,
                    region_number=region_number,
                    label=label,
                    width=width,
                    height=height,
                    corners=document.get("corners"),
                    found=True,
                    min_area_ratio=min_area_ratio,
                    order_corners=order_corners,
                )
            )

    if detections:
        return _drop_duplicate_regions(detections) if drop_duplicates else detections
    return [
        _detection_from_corners(
            page_number=page_number,
            region_number=1,
            label="full frame",
            width=width,
            height=height,
            corners=None,
            found=False,
        )
    ]


def _order_box_points(box: np.ndarray) -> np.ndarray:
    # The box isn't guaranteed to start at a fixed corner; order by summed/diffed coordinates so
    # the four points line up consistently with the destination rectangle below (smallest sum =
    # top-left, largest sum = bottom-right, etc).
    ordered = np.zeros((4, 2), dtype="float32")
    summed = box.sum(axis=1)
    ordered[0] = box[np.argmin(summed)]
    ordered[2] = box[np.argmax(summed)]
    diffed = np.diff(box, axis=1)
    ordered[1] = box[np.argmin(diffed)]
    ordered[3] = box[np.argmax(diffed)]
    return ordered


def crop_page(
    image_bgr: np.ndarray,
    detection: PageDetection,
    *,
    padding: int = 0,
    padding_ratio: float = 0.0,
    top_padding_ratio: float = 0.0,
) -> np.ndarray:
    if detection.method in ("fallback_full_frame", "fallback_invalid_response"):
        return image_bgr

    ordered = _order_box_points(np.array(detection.box, dtype="float32"))
    top_left, top_right, bottom_right, bottom_left = ordered
    target_width = max(int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))), 1)
    target_height = max(int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))), 1)
    # Scaled off the document's longest edge so the margin means the same thing on a 400px test
    # image and a 2200px PDF render, with the fixed pixel value as a floor for very small crops.
    # Deriving it from the longest edge (not each axis separately) gives the short axis the same
    # absolute margin as the long one — ID cards are wide, so their top/bottom edges are the ones
    # a proportional miss clips first.
    longest_edge = max(target_width, target_height)
    margin = int(round(max(float(padding), longest_edge * padding_ratio)))
    # The top edge is the one Gemini most consistently places too low, so it gets its own larger
    # ratio; it can never be tighter than the other three sides.
    top_margin = max(margin, int(round(longest_edge * top_padding_ratio)))
    # Mapping the detected quad to an inset rectangle (rather than growing the quad itself) makes
    # the warp sample the real pixels just outside the detected edge, so a slightly tight box from
    # Gemini doesn't shave off a border. Where the document sits against the frame edge there are
    # no pixels to recover and the margin fills with the border colour instead.
    destination = np.array(
        [
            [margin, top_margin],
            [margin + target_width - 1, top_margin],
            [margin + target_width - 1, top_margin + target_height - 1],
            [margin, top_margin + target_height - 1],
        ],
        dtype="float32",
    )
    # A perspective (not affine) transform, so this straightens skew/rotation and crops to the
    # document bounds in the same step.
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    # Cubic, not the default linear, so the warp keeps small print sharp.
    return cv2.warpPerspective(
        image_bgr, matrix, (target_width + 2 * margin, target_height + top_margin + margin), flags=cv2.INTER_CUBIC,
    )


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_upright(image_bgr: np.ndarray, rotation_degrees: float) -> np.ndarray:
    """Rotate clockwise by any angle Gemini reports. Exact quarter turns take cv2.rotate, which
    is a lossless transpose/flip; anything else needs a real affine rotation, with the canvas
    grown to the rotated bounding box so no corner of the document is cut off."""
    try:
        angle = float(rotation_degrees) % 360
    except (TypeError, ValueError):
        return image_bgr
    if angle == 0:
        return image_bgr
    quarter_turn = _ROTATIONS.get(int(angle)) if angle.is_integer() else None
    if quarter_turn is not None:
        return cv2.rotate(image_bgr, quarter_turn)

    height, width = image_bgr.shape[:2]
    # getRotationMatrix2D treats positive angles as counter-clockwise, so negate to rotate
    # clockwise as reported.
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    expanded_width = int(round(height * sin + width * cos))
    expanded_height = int(round(height * cos + width * sin))
    matrix[0, 2] += expanded_width / 2 - width / 2
    matrix[1, 2] += expanded_height / 2 - height / 2
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (expanded_width, expanded_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _resize_to_target(image_bgr: np.ndarray, target_long_edge: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if target_long_edge <= 0 or longest == 0:
        return image_bgr
    scale = target_long_edge / longest
    # Upscaling invents no real detail, so it is capped — a tiny crop is made legible, not
    # blown up into a soft, artefact-heavy image pretending to be high resolution.
    scale = min(scale, _MAX_UPSCALE)
    if abs(scale - 1.0) < 0.01:
        return image_bgr
    # Lanczos keeps letter edges crisper than cubic when a small crop is enlarged.
    interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image_bgr, (max(round(width * scale), 1), max(round(height * scale), 1)), interpolation=interpolation)


def _boost_contrast(image_bgr: np.ndarray, clip_limit: float) -> np.ndarray:
    if clip_limit <= 0:
        return image_bgr
    # CLAHE on lightness only, so contrast lifts in shadowed or unevenly lit scans without
    # shifting the document's colours (which matter for photo/hologram areas).
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((equalized, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _sharpen(image_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image_bgr
    # Unsharp mask: subtract a blurred copy to re-emphasise edges. Applied last, at final
    # resolution, so the sharpening isn't smeared away by a later resize.
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)


def enhance_crop(
    image_bgr: np.ndarray,
    rotation_degrees: int,
    *,
    target_long_edge: int = 0,
    contrast_clip: float = 0.0,
    sharpen_amount: float = 0.0,
) -> np.ndarray:
    # Order matters: rotate first so the size target applies to the final orientation, then
    # resize, then enhance at that final resolution.
    upright = rotate_upright(image_bgr, rotation_degrees)
    resized = _resize_to_target(upright, target_long_edge)
    return _sharpen(_boost_contrast(resized, contrast_clip), sharpen_amount)


def _crop_all(
    pairs: list[tuple[PageDetection, np.ndarray]],
    padding: int,
    padding_ratio: float,
    top_padding_ratio: float,
) -> list[np.ndarray]:
    return [
        crop_page(image, detection, padding=padding, padding_ratio=padding_ratio, top_padding_ratio=top_padding_ratio)
        for detection, image in pairs
    ]


def _finish_crops(
    pairs: list[tuple[PageDetection, np.ndarray]],
    raw_crops: list[np.ndarray],
    orientations: list[Any],
    target_long_edge: int,
    contrast_clip: float,
    sharpen_amount: float,
) -> list[PageCrop]:
    crops = []
    for (detection, _), raw_crop, orientation in zip(pairs, raw_crops, orientations):
        rotation, confidence, reason, top_edge = orientation_from_result(orientation)
        finished = enhance_crop(
            raw_crop,
            rotation,
            target_long_edge=target_long_edge,
            contrast_clip=contrast_clip,
            sharpen_amount=sharpen_amount,
        )
        crops.append(
            PageCrop(
                detection=detection,
                cropped_image=finished,
                rotation_degrees=rotation,
                rotation_confidence=confidence,
                rotation_reason=reason,
                rotation_top_edge=top_edge,
                raw_size=(raw_crop.shape[1], raw_crop.shape[0]),
            )
        )
    return crops


def log_rotations(crops: list[PageCrop], *, sha256: str, filename: str, document_type: str) -> None:
    for crop in crops:
        detection = crop.detection
        rotation = crop.rotation_degrees
        rotation_logger.info(
            json.dumps(
                {
                    **log_timestamps(),
                    "sha256": sha256,
                    "filename": filename,
                    "document_type": document_type,
                    "page_number": detection.page_number,
                    "region_number": detection.region_number,
                    "label": detection.label,
                    # What Gemini asked for.
                    "rotation_top_edge": crop.rotation_top_edge,
                    "rotation_requested_degrees": rotation,
                    "rotation_confidence": crop.rotation_confidence,
                    "rotation_reason": crop.rotation_reason,
                    # What OpenCV actually did with it.
                    "rotation_applied": rotation != 0,
                    "rotation_method": (
                        "none" if rotation == 0
                        else "cv2.rotate" if float(rotation).is_integer() and int(rotation) in _VALID_ROTATIONS
                        else "cv2.warpAffine"
                    ),
                    "box_angle_degrees": detection.angle_degrees,
                    "box_method": detection.method,
                    # Where on the page this crop came from, in the source image's own pixels.
                    # Ordered the way crop_page consumes it — top-left, top-right, bottom-right,
                    # bottom-left — so a logged quad can be read without re-deriving the order.
                    "crop_box": [[int(round(x)), int(round(y))] for x, y in _order_box_points(
                        np.array(detection.box, dtype="float32")
                    ).tolist()],
                    "crop_bbox": detection.axis_aligned_bbox,
                    "page_size": [detection.width, detection.height],
                    "size_before": list(crop.raw_size),
                    "size_after": [crop.cropped_image.shape[1], crop.cropped_image.shape[0]],
                }
            )
        )


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem or "document"
    return _UNSAFE_NAME.sub("_", stem)[:100] or "document"


def save_crop_outputs(
    *,
    output_dir: Path,
    filename: str,
    sha256: str,
    document_type: str,
    crops: list[PageCrop],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_stem(filename)}_{sha256[:8]}"
    single_crop = len(crops) == 1

    image_paths: list[Path] = []
    for crop in crops:
        suffix = "" if single_crop else f"_page{crop.detection.page_number}_doc{crop.detection.region_number}"
        image_path = output_dir / f"{stem}{suffix}.png"
        image_path.write_bytes(encode_png(crop.cropped_image))
        image_paths.append(image_path)

    pages: dict[int, dict[str, Any]] = {}
    for crop, image_path in zip(crops, image_paths):
        detection = crop.detection
        page = pages.setdefault(
            detection.page_number,
            {
                "page_number": detection.page_number,
                "width": detection.width,
                "height": detection.height,
                "documents_found": 0,
                "documents": [],
            },
        )
        document = asdict(detection)
        for key in ("page_number", "width", "height"):
            del document[key]
        page["documents"].append(
            {
                **document,
                "rotation_degrees": crop.rotation_degrees,
                "rotation_confidence": crop.rotation_confidence,
                "rotation_reason": crop.rotation_reason,
                "cropped_file": image_path.name,
            }
        )
        page["documents_found"] = len(page["documents"])

    metadata = {
        "source_filename": filename,
        "document_type": document_type,
        "sha256": sha256,
        "pages": list(pages.values()),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return image_paths


def save_merged_document(
    image_paths: list[Path], *, output_dir: Path, filename: str, sha256: str,
) -> Path | None:
    """Stitch a document's own final cropped/rotated PNGs into one downloadable PDF.

    Used only for a document built by merging two separately-uploaded, individually-incomplete
    files (one showing the front, the other the back) into one - the crop pipeline has already
    produced one clean, upright PNG per side above; this just combines them, in page order, into
    the single file the person who uploaded them actually wants to download. Named with the same
    stem convention `save_crop_outputs` already uses, so it sits right alongside that document's
    own crop PNGs and metadata JSON, discoverable the same way.
    """
    if not image_paths:
        return None
    stem = f"{_safe_stem(filename)}_{sha256[:8]}"
    pages = [Image.open(path).convert("RGB") for path in image_paths]
    merged_path = output_dir / f"{stem}_merged.pdf"
    pages[0].save(merged_path, format="PDF", save_all=True, append_images=pages[1:], **PDF_PAGE_QUALITY)
    return merged_path


async def run_detection(
    *,
    content: bytes,
    filename: str,
    mime_type: str,
    document_type: str,
    sha256: str,
    settings: Settings,
    gemini_service: DocumentVision,
    box_detection: bool = True,
    merged_national_id: bool = False,
) -> list[Path] | None:
    """Crops, rotates and enhances every document in a file and saves the results.

    `box_detection=False` is for a file that already holds exactly one document cut to its own edges
    (a type-detection cut-out): each page is taken whole, so no second detection call is made and
    nothing next to the document can be picked up again - only orientation and enhancement run.
    `merged_national_id`: a National ID's front and back sent as two files and merged - see `_card_box`."""
    # Rendering/encoding/cropping/saving are CPU-bound and offloaded to worker threads so they
    # never block the event loop; the Gemini call in between is awaited natively so it runs
    # concurrently with the main extraction call in app/services/ocr_pipeline.py, not serially after it.
    started_at = time.perf_counter()
    try:
        images = await asyncio.to_thread(render_pages, content, mime_type, settings.detection_pdf_dpi)
        encoded_pages = await asyncio.to_thread(_encode_pages_for_detection, images)
        page_results = (
            await gemini_service.detect_box_corners(encoded_pages) if box_detection
            else [{"found": False, "documents": []} for _ in images]
        )
        pairs = []
        for index, image in enumerate(images):
            detections = page_detections(
                page_number=index + 1, width=image.shape[1], height=image.shape[0], page_result=page_results[index],
                order_corners=merged_national_id,
            )
            pairs.extend((detection, image) for detection in detections)
        # Gemini says where the documents are; OpenCV snaps those regions onto the real edges so
        # the perspective warp deskews properly instead of introducing a shear of its own.
        pairs = await asyncio.to_thread(_refine_pairs, pairs)
        raw_crops = await asyncio.to_thread(
            _crop_all,
            pairs,
            settings.detection_crop_padding_px,
            settings.detection_crop_padding_ratio,
            settings.detection_crop_top_padding_ratio,
        )
        # Orientation is judged on the cropped documents, batched into a single call so the
        # number of documents never multiplies the number of requests. A failure here degrades
        # to "leave every crop as it is" rather than losing the crops altogether.
        encoded_crops = await asyncio.to_thread(_encode_pages_for_detection, raw_crops)
        try:
            orientations: list[Any] = await gemini_service.detect_orientations(encoded_crops)
        except Exception:
            logger.warning("Orientation detection failed sha256=%s; saving crops unrotated", sha256, exc_info=True)
            orientations = [None] * len(raw_crops)

        crops = await asyncio.to_thread(
            _finish_crops,
            pairs,
            raw_crops,
            orientations,
            settings.detection_crop_target_long_edge,
            settings.detection_crop_contrast_clip,
            settings.detection_crop_sharpen_amount,
        )
        log_rotations(crops, sha256=sha256, filename=filename, document_type=document_type)
        for crop in crops:
            confidence = crop.rotation_confidence
            if confidence is not None and confidence < settings.detection_rotation_min_confidence:
                logger.warning(
                    "Low-confidence rotation sha256=%s page=%d doc=%d rotation=%s confidence=%.2f reason=%s",
                    sha256, crop.detection.page_number, crop.detection.region_number,
                    crop.rotation_degrees, confidence, crop.rotation_reason,
                )
        saved_paths = await asyncio.to_thread(
            save_crop_outputs,
            output_dir=Path(settings.detection_output_dir),
            filename=filename,
            sha256=sha256,
            document_type=document_type,
            crops=crops,
        )
        logger.info(
            "Detection crop saved sha256=%s pages=%d paths=%s duration_sec=%.2f",
            sha256, len(crops), [str(path) for path in saved_paths], time.perf_counter() - started_at,
        )
        return saved_paths
    except Exception:
        logger.exception("Document box detection/crop failed sha256=%s", sha256)
        return None
