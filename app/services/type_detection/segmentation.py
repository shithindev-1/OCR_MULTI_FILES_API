"""Splitting one uploaded file into the documents it holds - for the type-detection endpoint.

The caller sends no document type, so before anything else each file is looked at once as a whole
(`GeminiService.segment_documents`): which documents are in it, of what type, on which pages, and
where. Regions of the same physical document - a card's front and back, a contract's pages - are
grouped into one document.

A file holding a single document is passed on as it is, untouched: cutting it out of its own page
could only lose something. Only a file holding several documents is cut up, each document becoming
one new file - a JPG for a single region, a PDF with a page per region otherwise - which the normal
pipeline then reads, verifies, crops and rotates exactly like an upload.
"""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.schemas.ocr import DETECT_DOCUMENT_TYPES, OTHER_DOCUMENT_TYPE
from app.services import result_cache
from app.services.detection import (
    PDF_PAGE_QUALITY,
    PageDetection,
    crop_page,
    downscale_for_detection,
    encode_png,
    page_detections,
    refine_detection,
    render_pages,
)

logger = logging.getLogger("uae_ocr")

OTHER_TYPE = OTHER_DOCUMENT_TYPE
# Pages are rendered sharper than for the crop/rotate output (detection_pdf_dpi): a cut-out ID card
# is read again from these pixels, and its long numbers need the resolution.
_SEGMENT_PDF_DPI = 300
_SIDE_ORDER = {"front": 0, "single": 1, "back": 2}
# What replaces the part of a document that another document was lying on (light grey, BGR).
COVER_FILL = (230, 230, 230)
# Smallest share of the page a split region may cover: an ID card on an A4 scan is under 4% of it, so
# the box call's 5% sanity limit would turn it into the whole page.
_MIN_REGION_AREA_RATIO = 0.005
# Margin kept around a cut-out on a page shared with other documents, as a share of its longest edge:
# a split box placed right on a card's edge would otherwise cut the edge off. Other documents inside
# the margin are painted out, and the final crop trims the cut-out to the document's own corners.
_SHARED_PAGE_MARGIN_RATIO = 0.02
# Two regions of the SAME document and side overlapping this much are one region reported twice.
_DUPLICATE_OVERLAP = 0.8


class DocumentSegmenter(Protocol):
    async def segment_documents(self, page_images_png: list[bytes]) -> list[dict[str, Any]]: ...


@dataclass
class Segment:
    """One document found in a file."""

    document_type: str  # one of DETECT_DOCUMENT_TYPES, or "other"
    pages: list[int]  # 1-based pages of the original file it was found on
    # The document's own file, cut out of the original - or None when the original file holds this
    # one document only and is used as it is.
    content: bytes | None = None
    extension: str = ""  # ".jpg" or ".pdf" when content is set
    regions: list[PageDetection] = field(default_factory=list, repr=False)
    # Share of each region (in order) blanked because another document lay on top of it.
    covered: list[float] = field(default_factory=list)
    # For an "other" document: its exact name as the split call read it ("UAE driving licence"), else "".
    detected_type: str = ""


def _region_label(region: dict[str, Any], page_number: int, position: int) -> str:
    """Carries a region's type, document, side and the documents lying on top of it through
    `page_detections`, which keeps only a label per region: "<type>:<document key>:<side>:<covered by>"."""
    document_type = str(region.get("document_type") or OTHER_TYPE)
    if document_type not in DETECT_DOCUMENT_TYPES:
        document_type = OTHER_TYPE
    try:
        number = int(region.get("document_number"))
    except (TypeError, ValueError):
        # No usable number: its own document, never merged into another.
        number = -(page_number * 1000 + position)
    side = str(region.get("side") or "single")
    # Its own number is kept: it means the other side of the same card lies on top of this region.
    covered_by = []
    listed = region.get("covered_by")
    for value in listed if isinstance(listed, list) else []:
        try:
            covered_by.append(str(int(value)))
        except (TypeError, ValueError):
            continue
    return f"{document_type}:{number}:{side if side in _SIDE_ORDER else 'single'}:{','.join(covered_by)}"


def _document_name(region: dict[str, Any]) -> str:
    """An unsupported document's own name from the split answer - plain text, kept short."""
    name = " ".join(str(region.get("document_name") or "").split())
    return name[:80]


def parse_side(label: str) -> str:
    """A region's side ("front", "back" or "single") from its detection label - "" for any other label."""
    parsed = _parse_label(label)
    return parsed[2] if parsed else ""


def _parse_label(label: str) -> tuple[str, int, str, set[int]] | None:
    parts = label.split(":")
    if len(parts) != 4:
        return None  # "full frame" - the fallback for a page where nothing was found
    try:
        return parts[0], int(parts[1]), parts[2], {int(value) for value in parts[3].split(",") if value}
    except ValueError:
        return None


def _crops_to_file(crops: list[np.ndarray], pdf_dpi: int) -> tuple[bytes, str]:
    """One crop -> a lossless PNG; several -> one PDF with a page per crop.

    Every cut-out is read again and then cropped once more, so it must lose nothing: a JPEG cut-out
    or a PDF rendered back at a lower DPI than it was written at softened small print into smudge
    before the final crop even started. The PDF is written at `pdf_dpi` - the DPI the crop step
    renders PDFs at - so its pages come back at exactly their own pixel size.
    """
    if len(crops) == 1:
        ok, buffer = cv2.imencode(".png", crops[0])
        if not ok:
            raise ValueError("Failed to encode document crop as PNG")
        return buffer.tobytes(), ".png"
    pages = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
    buffer = BytesIO()
    pages[0].save(
        buffer, format="PDF", save_all=True, append_images=pages[1:], resolution=pdf_dpi, **PDF_PAGE_QUALITY,
    )
    return buffer.getvalue(), ".pdf"


def _fractions(corners: Any) -> Any:
    """The corners as fractions of the image (0 to 1).

    The prompt asks for fractions, but the model sometimes answers on Gemini's own 0-1000 box scale
    instead. Read as fractions, every such box falls outside the image, becomes the whole page, and
    the boxes then collapse into one - a passport, an ID and a visa in one photo came back as a
    single document. A box with any value above 1 is therefore read as 0-1000 and scaled down.
    """
    if not isinstance(corners, list):
        return corners
    try:
        values = [float(value) for corner in corners for value in (corner["x"], corner["y"])]
    except (KeyError, TypeError, ValueError):
        return corners
    if not values or max(values) <= 1.0 or max(values) > 1000.0:
        return corners
    return [{"x": float(corner["x"]) / 1000.0, "y": float(corner["y"]) / 1000.0} for corner in corners]


def _region_corners(region: dict[str, Any]) -> Any:
    """The region's four corners as image fractions, from `box_2d` ([ymin, xmin, ymax, xmax] on
    0-1000) - or, from an answer that gave `corners`, those (see `_fractions`)."""
    box = region.get("box_2d")
    if isinstance(box, list) and len(box) == 4:
        try:
            ymin, xmin, ymax, xmax = (min(max(float(value), 0.0), 1000.0) / 1000.0 for value in box)
        except (TypeError, ValueError):
            return None
        return [{"x": xmin, "y": ymin}, {"x": xmax, "y": ymin}, {"x": xmax, "y": ymax}, {"x": xmin, "y": ymax}]
    return _fractions(region.get("corners"))


@dataclass
class _Region:
    detection: PageDetection
    image: np.ndarray
    document_type: str
    number: int
    side: str
    covered_by: set[int]
    name: str = ""


def _quad(detection: PageDetection) -> np.ndarray:
    return np.array(detection.box, dtype=np.float32)


def _overlap(first: PageDetection, second: PageDetection) -> tuple[float, np.ndarray | None]:
    """Area and outline of the part two regions' outlines share (both are convex four-sided shapes)."""
    area, polygon = cv2.intersectConvexConvex(_quad(first), _quad(second))
    return float(area), (polygon if polygon is not None and area > 0 else None)


def _isolate(target: _Region, page_regions: list[_Region]) -> tuple[np.ndarray, float]:
    """The page with every other document painted COVER_FILL, and the share of `target` that was painted.

    Another document is painted wherever it lies - inside `target`'s outline and in the margin around
    it - except where `target` lies on top of it (the other region lists it in covered_by and is not
    itself listed by `target`): there `target`'s own pixels are kept. When the model gives no order,
    painting the shared part is the safe choice: the cut-out must never show a piece of another document.

    The two sides of one card (same document number, photographed separately) are handled the same
    way: a side whose covered_by lists its own number has the other side lying on top of it, and
    that part is painted; the side on top is left whole. A region that fell back to the whole page is
    no real outline and paints nothing.
    """
    image = target.image
    painted = 0.0
    target_quad = _quad(target.detection).astype(np.int32)
    for other in page_regions:
        if other is target or other.detection.method.startswith("fallback"):
            continue
        keep_shared_part = False
        if other.number == target.number:
            # The other side of the same card: its shared part is painted only when it lies on top.
            keep_shared_part = (
                other.side == target.side or target.number not in target.covered_by
                or other.number in other.covered_by  # both claim to be covered - no usable order
            )
        else:
            keep_shared_part = target.number in other.covered_by and other.number not in target.covered_by
        area, polygon = _overlap(target.detection, other.detection)
        mask = np.zeros(target.image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, _quad(other.detection).astype(np.int32), 255)
        if keep_shared_part:
            cv2.fillConvexPoly(mask, target_quad, 0)
        elif polygon is not None:
            painted += area
        if not mask.any():
            continue
        if image is target.image:
            image = target.image.copy()
        image[mask > 0] = COVER_FILL
    total = max(cv2.contourArea(_quad(target.detection)), 1.0)
    return image, min(painted / total, 1.0)


def _drop_repeats(regions: list[_Region]) -> list[_Region]:
    """Drops a region only when it repeats one already kept for the same document and side - two
    different documents are never merged, however much they overlap."""
    kept: list[_Region] = []
    for region in regions:
        repeat = any(
            other.number == region.number and other.side == region.side
            and _overlap(region.detection, other.detection)[0]
            >= _DUPLICATE_OVERLAP * min(cv2.contourArea(_quad(region.detection)), cv2.contourArea(_quad(other.detection)))
            for other in kept
        )
        if repeat:
            logger.warning("Segmentation region repeated and dropped page=%d document=%d side=%s",
                           region.detection.page_number, region.number, region.side)
        else:
            kept.append(region)
    return kept


def build_segments(
    images: list[np.ndarray], page_results: list[dict[str, Any]], settings: Settings,
) -> list[Segment]:
    """Turns the segmentation answer into one `Segment` per physical document, in the order each
    first appears. Regions of one document are ordered page by page, a front before a back.

    A file holding several documents is cut up so that each cut-out shows its own document only: the
    region's exact outline (snapped to the document's real edges), no margin on a page shared with
    other documents, and any part another document was lying on painted light grey.
    """
    regions_by_page: list[list[_Region]] = []
    for index, (image, page_result) in enumerate(zip(images, page_results)):
        page_number = index + 1
        # Anything but a list of region objects is read as "nothing found on this page".
        regions = page_result.get("regions") if isinstance(page_result, dict) else None
        raw = [region for region in regions if isinstance(region, dict)] if isinstance(regions, list) else []
        detections = page_detections(
            page_number=page_number, width=image.shape[1], height=image.shape[0],
            page_result={
                "found": bool(raw),
                "documents": [
                    {"label": _region_label(region, page_number, position), "corners": _region_corners(region)}
                    for position, region in enumerate(raw)
                ],
            },
            drop_duplicates=False,
            min_area_ratio=_MIN_REGION_AREA_RATIO,
        )
        page_regions: list[_Region] = []
        for detection in detections:
            parsed = _parse_label(detection.label)
            if parsed is None:
                continue  # a blank page
            siblings = [other.box for other in detections if other is not detection]
            document_type, number, side, covered_by = parsed
            # page_detections keeps the answer's order: region_number N is the answer's Nth region.
            name = _document_name(raw[detection.region_number - 1]) if document_type == OTHER_TYPE else ""
            page_regions.append(_Region(
                refine_detection(image, detection, siblings), image, document_type, number, side, covered_by, name,
            ))
        regions_by_page.append(_drop_repeats(page_regions))

    groups: dict[int, list[_Region]] = {}
    for page_regions in regions_by_page:
        for region in page_regions:
            groups.setdefault(region.number, []).append(region)

    segments: list[Segment] = []
    for members in groups.values():
        members.sort(key=lambda region: (region.detection.page_number, _SIDE_ORDER[region.side], region.detection.region_number))
        # Regions of one document should all carry its type; if the model wavered, the majority wins.
        document_type = Counter(region.document_type for region in members).most_common(1)[0][0]
        segments.append(Segment(
            document_type=document_type,
            pages=sorted({region.detection.page_number for region in members}),
            regions=[region.detection for region in members],
            detected_type=next((region.name for region in members if region.name), "") if document_type == OTHER_TYPE else "",
        ))

    if len(segments) > 1:
        for segment, members in zip(segments, groups.values()):
            crops = []
            for region in members:
                page_regions = regions_by_page[region.detection.page_number - 1]
                isolated, covered = _isolate(region, page_regions)
                shared_page = len(page_regions) > 1
                crops.append(crop_page(
                    isolated, region.detection,
                    padding=0 if shared_page else settings.detection_crop_padding_px,
                    padding_ratio=_SHARED_PAGE_MARGIN_RATIO if shared_page else settings.detection_crop_padding_ratio,
                    top_padding_ratio=0.0 if shared_page else settings.detection_crop_top_padding_ratio,
                ))
                segment.covered.append(round(covered, 3))
            segment.content, segment.extension = _crops_to_file(crops, settings.detection_pdf_dpi)
    return segments


async def segment_file(
    *, content: bytes, mime_type: str, sha256: str, settings: Settings, gemini_service: DocumentSegmenter,
) -> list[Segment]:
    """Every document in one file. A file where nothing at all was found comes back as a single
    "other" document, so it is still reported. Raises whatever the model call raises."""
    images = await asyncio.to_thread(render_pages, content, mime_type, max(settings.detection_pdf_dpi, _SEGMENT_PDF_DPI))
    cache_key = ("segmentation", settings.gemini_model, sha256)
    page_results = result_cache.get(cache_key)
    fresh = False
    if page_results is None:
        encoded = await asyncio.to_thread(
            lambda: [encode_png(downscale_for_detection(image, settings.gemini_max_image_edge)) for image in images]
        )
        page_results = await gemini_service.segment_documents(encoded)
        fresh = True
        logger.info(
            "Segmentation answer sha256=%s regions=%s", sha256,
            [
                (page_number, region.get("document_type"), region.get("side"), region.get("document_number"),
                 region.get("box_2d") or region.get("corners"))
                for page_number, page in enumerate(page_results, start=1)
                for region in (page or {}).get("regions") or [] if isinstance(region, dict)
            ],
        )
    segments = await asyncio.to_thread(build_segments, images, page_results, settings)
    if fresh:
        # Cached only once it was usable, so a retry after a failure asks the model again instead of
        # reusing the answer that failed.
        result_cache.put(cache_key, page_results)
    logger.info(
        "Segmentation sha256=%s pages=%d documents=%s covered=%s",
        sha256, len(images), [(segment.document_type, segment.pages) for segment in segments],
        [segment.covered for segment in segments],
    )
    return segments or [Segment(document_type=OTHER_TYPE, pages=list(range(1, len(images) + 1)))]
