"""Where a request's files go: local output folders, the processed (cropped/rotated/merged)
result uploaded beside each original in blob storage, and reading such a result back."""

import asyncio
import posixpath
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image

from app.core.config import settings
from app.services import blob_storage
from app.services.blob_storage import BlobStorageError
from app.services.leasing.sources import safe_name
from app.services.post_processing import document_label
from app.services.validation import ALLOWED_FILES

# Pillow format names for an _ocr result that keeps its original's image type.
_IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


def output_root() -> Path:
    return Path(settings.leasing_output_dir)


def document_folder(refnumber: str, document_id: str, document_name: str) -> Path:
    folder = output_root() / safe_name(refnumber) / f"{safe_name(document_id)}_{safe_name(document_name)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def merged_folder(refnumber: str, front_id: str, back_id: str) -> Path:
    folder = output_root() / safe_name(refnumber) / f"merged_{safe_name(front_id)}_{safe_name(back_id)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _ocr_output(image_paths: list[Path], original_extension: str) -> tuple[bytes, str, str]:
    """Builds the uploaded result from a document's final cropped/rotated PNGs.

    Returns (content, extension, content_type). One crop keeps the original's own image format
    (a .jpg stays a .jpg); several crops - a PDF's pages, or two cards found in one photo - become
    one PDF, a page per crop, since a single image file cannot hold more than one.
    """
    images = [Image.open(path) for path in image_paths]
    buffer = BytesIO()
    image_format = _IMAGE_FORMATS.get(original_extension)
    if len(images) == 1 and image_format:
        image = images[0].convert("RGB") if image_format == "JPEG" else images[0]
        image.save(buffer, format=image_format, **({"quality": 95} if image_format == "JPEG" else {}))
        return buffer.getvalue(), original_extension, ALLOWED_FILES[original_extension]
    pages = [image.convert("RGB") for image in images]
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue(), ".pdf", "application/pdf"


def _ocr_blob_path(blob_path: str, extension: str) -> str:
    """ApplicationFiles/.../Passport.jpg -> ApplicationFiles/.../Passport_ocr.jpg (same folder)."""
    folder, _, name = blob_path.rpartition("/")
    renamed = f"{posixpath.splitext(name)[0]}_ocr{extension}"
    return f"{folder}/{renamed}" if folder else renamed


def type_file_stem(document_type: str) -> str:
    """The processed file's name for a type-detected document: national_id -> "National_ID",
    trade_license -> "Trade_Licence"."""
    words = document_label(document_type).split()
    return "_".join(word if word.isupper() else word.capitalize() for word in words)


async def upload_ocr_result(
    blob_path: str, crop_paths: list[Path], merged_pdf: Path | None, name: str | None = None,
) -> str:
    """Uploads one document's result beside its original and returns its full blob URL. It is
    named `<original>_ocr.<ext>`, or `<name>.<ext>` when a name is given."""
    storage = blob_storage.get_blob_storage()
    if storage is None:
        raise BlobStorageError("blob storage is not configured")
    if merged_pdf is not None:
        content, extension, content_type = merged_pdf.read_bytes(), ".pdf", "application/pdf"
    else:
        extension = posixpath.splitext(blob_path)[1].lower()
        content, extension, content_type = await asyncio.to_thread(_ocr_output, crop_paths, extension)
    if name:
        folder = posixpath.dirname(blob_path)
        target = posixpath.join(folder, f"{name}{extension}") if folder else f"{name}{extension}"
    else:
        target = _ocr_blob_path(blob_path, extension)
    await storage.upload(target, content, content_type)
    return storage.url(target)


_OCR_FILE_NAME = re.compile(r"_ocr\.(jpe?g|png|webp|pdf)$", re.IGNORECASE)


# A merged or multi-page result is a PDF of full-resolution crops - allow well past the upload limit.
_MAX_OCR_FILE_BYTES = 100 * 1024 * 1024


async def fetch_ocr_file(ocr_file: str) -> tuple[bytes, str, str]:
    """Reads one uploaded `<name>_ocr.<ext>` result back out of blob storage, for viewing.

    `ocr_file` is either the full blob URL a response's `ocr_file` carries, or the blob path
    inside the container. Returns (content, content_type, filename). Only `_ocr` results are
    served - never an original upload or any other blob in the container.
    """
    storage = blob_storage.get_blob_storage()
    if storage is None:
        raise HTTPException(status_code=503, detail="Viewing processed files is not available right now.")
    path = storage.blob_path(ocr_file)
    if path is None:
        raise HTTPException(status_code=400, detail="This file cannot be viewed here. Please use the ocr_file link from the response.")
    filename = posixpath.basename(path)
    if ".." in path.split("/") or not _OCR_FILE_NAME.search(filename):
        raise HTTPException(status_code=400, detail="Only processed files (ending in _ocr) can be viewed here.")
    try:
        content, _ = await storage.download(path, max_bytes=_MAX_OCR_FILE_BYTES)
    except BlobStorageError as exc:
        raise HTTPException(
            status_code=404 if exc.not_found else 502,
            detail="We could not find this file." if exc.not_found else
                   "We could not open this file right now. Please try again in a few minutes.",
        ) from exc
    return content, ALLOWED_FILES[posixpath.splitext(filename)[1].lower()], filename


def assign_type_file_names(
    documents: dict[tuple[int, ...], dict[str, Any]], items: list[dict[str, Any]],
    tenant_numbers: dict[tuple[int, ...], list[int]] | None = None,
) -> None:
    """Names each processed file after its type - National_ID, then National_ID_2 for a second one
    in the same folder - in upload order. A name that an original in that folder already has gets
    "_ocr" added, so an original is never overwritten.

    `tenant_numbers` - given when the request holds several tenants - puts the document's tenant
    first: Tenant_1_National_ID, Tenant_2_Passport. A document shared by several tenants (a trade
    licence with more than one listed signatory) carries every number: Tenant_1_2_Trade_Licence."""
    originals: dict[str, set[str]] = {}
    for item in items:
        path = item.get("source_blob_path") or item.get("blob_path")
        if path:
            folder, name = posixpath.split(path)
            originals.setdefault(folder, set()).add(posixpath.splitext(name)[0].lower())
    used: dict[tuple[str, str], int] = {}
    for key in sorted(documents):
        path = items[key[0]].get("blob_path")
        if not path:
            continue
        folder = posixpath.dirname(path)
        stem = type_file_stem(documents[key]["document_type"])
        if tenant_numbers and tenant_numbers.get(key):
            stem = "Tenant_" + "_".join(str(number) for number in tenant_numbers[key]) + f"_{stem}"
        count = used[(folder, stem)] = used.get((folder, stem), 0) + 1
        name = stem if count == 1 else f"{stem}_{count}"
        if name.lower() in originals.get(folder, set()):
            name = f"{name}_ocr"
        documents[key]["_ocr_name"] = name
