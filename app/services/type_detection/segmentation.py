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
from app.schemas.ocr import VERIFIABLE_TYPES
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

OTHER_TYPE = "other"
# Pages are rendered sharper than for the crop/rotate output (detection_pdf_dpi): a cut-out ID card
# is read again from these pixels, and its long numbers need the resolution.
_SEGMENT_PDF_DPI = 300
_SIDE_ORDER = {"front": 0, "single": 1, "back": 2}


class DocumentSegmenter(Protocol):
    async def segment_documents(self, page_images_png: list[bytes]) -> list[dict[str, Any]]: ...


@dataclass
class Segment:
    """One document found in a file."""

    document_type: str  # one of VERIFIABLE_TYPES, or "other"
    pages: list[int]  # 1-based pages of the original file it was found on
    # The document's own file, cut out of the original - or None when the original file holds this
    # one document only and is used as it is.
    content: bytes | None = None
    extension: str = ""  # ".jpg" or ".pdf" when content is set
    regions: list[PageDetection] = field(default_factory=list, repr=False)


def _region_label(region: dict[str, Any], page_number: int, position: int) -> str:
    """Carries a region's type, document and side through `page_detections`, which keeps only a
    label per region: "<type>:<document key>:<side>"."""
    document_type = str(region.get("document_type") or OTHER_TYPE)
    if document_type not in VERIFIABLE_TYPES:
        document_type = OTHER_TYPE
    try:
        number = int(region.get("document_number"))
    except (TypeError, ValueError):
        # No usable number: its own document, never merged into another.
        number = -(page_number * 1000 + position)
    side = str(region.get("side") or "single")
    return f"{document_type}:{number}:{side if side in _SIDE_ORDER else 'single'}"


def _parse_label(label: str) -> tuple[str, int, str] | None:
    parts = label.split(":")
    if len(parts) != 3:
        return None  # "full frame" - the fallback for a page where nothing was found
    try:
        return parts[0], int(parts[1]), parts[2]
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


def build_segments(
    images: list[np.ndarray], page_results: list[dict[str, Any]], settings: Settings,
) -> list[Segment]:
    """Turns the segmentation answer into one `Segment` per physical document, in the order each
    first appears. Regions of one document are ordered page by page, a front before a back."""
    groups: dict[int, list[tuple[PageDetection, np.ndarray, str, str]]] = {}
    for index, (image, page_result) in enumerate(zip(images, page_results)):
        page_number = index + 1
        regions = [region for region in (page_result or {}).get("regions") or [] if isinstance(region, dict)]
        detections = page_detections(
            page_number=page_number, width=image.shape[1], height=image.shape[0],
            page_result={
                "found": bool(regions),
                "documents": [
                    {"label": _region_label(region, page_number, position), "corners": _region_corners(region)}
                    for position, region in enumerate(regions)
                ],
            },
        )
        for detection in detections:
            parsed = _parse_label(detection.label)
            if parsed is None:
                continue  # a blank page
            siblings = [other.box for other in detections if other is not detection]
            refined = refine_detection(image, detection, siblings)
            document_type, number, side = parsed
            groups.setdefault(number, []).append((refined, image, document_type, side))

    segments: list[Segment] = []
    for members in groups.values():
        members.sort(key=lambda member: (member[0].page_number, _SIDE_ORDER[member[3]], member[0].region_number))
        # Regions of one document should all carry its type; if the model wavered, the majority wins.
        document_type = Counter(member[2] for member in members).most_common(1)[0][0]
        segments.append(Segment(
            document_type=document_type,
            pages=sorted({member[0].page_number for member in members}),
            regions=[member[0] for member in members],
        ))

    if len(segments) > 1:
        for segment, members in zip(segments, groups.values()):
            crops = [
                crop_page(
                    image, detection,
                    padding=settings.detection_crop_padding_px,
                    padding_ratio=settings.detection_crop_padding_ratio,
                    top_padding_ratio=settings.detection_crop_top_padding_ratio,
                )
                for detection, image, _, _ in members
            ]
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
    if page_results is None:
        encoded = await asyncio.to_thread(
            lambda: [encode_png(downscale_for_detection(image, settings.gemini_max_image_edge)) for image in images]
        )
        page_results = await gemini_service.segment_documents(encoded)
        result_cache.put(cache_key, page_results)
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
    logger.info(
        "Segmentation sha256=%s pages=%d documents=%s",
        sha256, len(images), [(segment.document_type, segment.pages) for segment in segments],
    )
    return segments or [Segment(document_type=OTHER_TYPE, pages=list(range(1, len(images) + 1)))]
