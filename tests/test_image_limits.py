import io

import pytest
from PIL import Image

from app.services.gemini import downscale_image_for_api

# The API rejects anything with an edge above this outright.
API_HARD_LIMIT = 8000


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 180, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _size(content: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as image:
        return image.size


def test_oversized_upload_is_brought_under_the_api_limit() -> None:
    # Reproduces the 400: "image dimensions exceed max allowed size: 8000 pixels".
    original = _png(9000, 3000)
    assert max(_size(original)) > API_HARD_LIMIT

    content, media_type = downscale_image_for_api(original, "image/png", 1568)

    width, height = _size(content)
    assert max(width, height) == 1568
    assert max(width, height) < API_HARD_LIMIT
    # Re-encoded as JPEG, so the declared type has to follow the bytes.
    assert media_type == "image/jpeg"


def test_downscaling_preserves_aspect_ratio() -> None:
    content, _ = downscale_image_for_api(_png(4000, 2000), "image/png", 1000)

    width, height = _size(content)
    assert (width, height) == (1000, 500)


@pytest.mark.parametrize("size", [(1568, 800), (800, 1568), (400, 300)])
def test_images_within_the_limit_are_passed_through_untouched(size) -> None:
    original = _png(*size)

    content, media_type = downscale_image_for_api(original, "image/png", 1568)

    assert content is original
    assert media_type == "image/png"


def test_pdfs_are_never_touched() -> None:
    pdf_bytes = b"%PDF-1.4 not really a pdf"

    content, media_type = downscale_image_for_api(pdf_bytes, "application/pdf", 1568)

    assert content is pdf_bytes
    assert media_type == "application/pdf"


def test_undecodable_bytes_fall_through_rather_than_failing_extraction() -> None:
    junk = b"not an image at all"

    content, media_type = downscale_image_for_api(junk, "image/png", 1568)

    assert content is junk
    assert media_type == "image/png"


def test_zero_max_edge_disables_downscaling() -> None:
    original = _png(9000, 3000)

    content, media_type = downscale_image_for_api(original, "image/png", 0)

    assert content is original
    assert media_type == "image/png"
