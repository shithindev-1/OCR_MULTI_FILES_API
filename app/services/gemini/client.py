"""The Gemini client: extraction, box detection, orientation, segmentation and the
verification calls."""

import asyncio
import json
import logging
import time
import uuid
from datetime import date
from functools import lru_cache
from io import BytesIO
from typing import Any

from google import genai
from google.genai import types
from PIL import Image, ImageOps

from app.core.config import Settings, get_settings
from app.core.logging import log_timestamps
from app.schemas.ocr import COMPANY_EXTRACTION_FIELDS, DATA_FIELDS, is_company
from app.services.gemini.prompts import (
    BOX_DETECTION_PROMPT,
    BOX_DETECTION_SCHEMA,
    COMPANY_CROSS_VERIFICATION_PROMPT,
    COMPANY_SYSTEM_INSTRUCTION,
    COUNTRY_CODE_PROMPT,
    COUNTRY_CODE_SCHEMA,
    CROSS_VERIFICATION_PROMPT,
    DOCUMENT_TYPE_CLASSIFICATION_INSTRUCTION,
    EXTRACTION_PROMPT,
    ISSUES_SCHEMA,
    ORIENTATION_PROMPT,
    ORIENTATION_SCHEMA,
    PAIR_CONFIRMATION_PROMPT,
    PAIR_CONFIRMATION_SCHEMA,
    SEGMENTATION_PROMPT,
    SEGMENTATION_SCHEMA,
    SELF_VERIFICATION_PROMPT,
    SHORT_CROSS_MESSAGES,
    SHORT_SELF_MESSAGES,
    SYSTEM_INSTRUCTION,
)
from app.services.gemini.usage import (
    CALL_CROP,
    CALL_EXTRACTION,
    CALL_ROTATION,
    CALL_SEGMENTATION,
    CALL_VALIDATION,
    record_usage,
)
from app.services.post_processing import parse_extraction_text
from app.services.validation import log_processing

logger = logging.getLogger("uae_ocr")


hits_logger = logging.getLogger("uae_ocr.hits")


results_logger = logging.getLogger("uae_ocr.results")


token_breakdown_logger = logging.getLogger("uae_ocr.token_breakdown")


class GeminiServiceError(Exception):
    pass


def downscale_image_for_api(content: bytes, media_type: str, max_edge: int) -> tuple[bytes, str]:
    """Shrink an oversized upload before it is sent for extraction.

    Gemini bills images in 768x768 tiles and scales anything larger than its own maximum down
    before reading it, so sending a huge original buys no extra detail — it just wastes upload
    bandwidth and pushes the inline request payload towards the 20MB ceiling above which the
    Files API becomes mandatory. PDFs are passed through untouched; their pages are rasterised
    server-side and are not subject to this pixel limit.
    """
    if media_type == "application/pdf" or max_edge <= 0:
        return content, media_type
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            if max(width, height) <= max_edge:
                return content, media_type
            scale = max_edge / max(width, height)
            resized = ImageOps.exif_transpose(image).convert("RGB").resize(
                (max(round(width * scale), 1), max(round(height * scale), 1)), Image.LANCZOS
            )
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=92)
        logger.info(
            "Downscaled oversized upload for extraction from %dx%d to %dx%d",
            width, height, resized.width, resized.height,
        )
        # Re-encoded, so the declared media type has to follow the new bytes.
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        # Never block extraction on a resize problem; the original bytes may still be accepted.
        logger.warning("Could not downscale upload for extraction", exc_info=True)
        return content, media_type


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000),
        )
        self._system_tokens: dict[str, int] = {}
        self._system_tokens_lock = asyncio.Lock()

    @property
    def _vision_model(self) -> str:
        """Model for box detection and orientation - simple visual calls that a faster, cheaper
        model handles well. Extraction and verification always use GEMINI_MODEL."""
        return self.settings.gemini_vision_model or self.settings.gemini_model

    async def warm_up(self) -> None:
        """Opens the connection to Gemini before the first request needs it.

        A model lookup is free and carries no document; it only pays the one-off DNS/TLS setup that
        the first real call would otherwise add to its own latency.
        """
        for model in dict.fromkeys((self.settings.gemini_model, self._vision_model)):
            await self.client.aio.models.get(model=model)

    def _generation_config(self, **overrides: Any) -> types.GenerateContentConfig:
        # Every call here is a precise reading task, not a creative one. A low temperature and a
        # fixed seed keep the model close to its single most confident reading of the page instead
        # of sampling a less certain alternative - the difference between reliably transcribing a
        # passport number that happens to be printed on an Ejari certificate and skipping it because
        # a less-likely token won the sample. Gemini 2.5 also thinks adaptively by default, which can
        # burn the whole output budget before emitting an answer; a thinking_budget of 0 switches
        # that off on the Flash tier (Pro clamps to its own minimum and keeps thinking regardless,
        # which is a reason to prefer Flash for this workload).
        defaults: dict[str, Any] = {
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
            "temperature": 0.0,
            "seed": 0,
        }
        defaults.update(overrides)
        return types.GenerateContentConfig(**defaults)

    async def _get_system_tokens(self, system_instruction: str) -> int | None:
        # Each system instruction is a fixed constant, so it only needs measuring once per process:
        # count the same placeholder with and without it attached and take the delta, isolating
        # the instruction's own cost from the per-message wrapper overhead.
        if system_instruction in self._system_tokens:
            return self._system_tokens[system_instruction]
        async with self._system_tokens_lock:
            if system_instruction in self._system_tokens:
                return self._system_tokens[system_instruction]
            try:
                placeholder = "x"
                baseline, with_system = await asyncio.gather(
                    self.client.aio.models.count_tokens(model=self.settings.gemini_model, contents=placeholder),
                    self.client.aio.models.count_tokens(
                        model=self.settings.gemini_model,
                        contents=placeholder,
                        config=types.CountTokensConfig(system_instruction=system_instruction),
                    ),
                )
                self._system_tokens[system_instruction] = with_system.total_tokens - baseline.total_tokens
            except Exception:
                logger.warning("Could not measure system instruction token count", exc_info=True)
                return None
            return self._system_tokens[system_instruction]

    @staticmethod
    def _response_text(response: Any) -> str:
        """Concatenate the answer parts, skipping thought parts.

        The SDK's response.text shortcut warns or raises when a candidate carries non-text parts,
        so the parts are walked explicitly - which the extraction path needs anyway, to tell an
        empty answer apart from a blocked or truncated one.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        return "".join(
            part.text for part in parts if getattr(part, "text", None) and not getattr(part, "thought", False)
        )

    @staticmethod
    def _finish_details(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        return f"finish_reason={finish_reason}, block_reason={block_reason}"

    @staticmethod
    def _image_sizes(images_png: list[bytes]) -> list[str]:
        """"WxH" per image, read from the PNG header only - no decoding."""
        sizes = []
        for png in images_png:
            try:
                with Image.open(BytesIO(png)) as image:
                    sizes.append(f"{image.width}x{image.height}")
            except Exception:
                sizes.append("unknown")
        return sizes

    def _log_vision_call(
        self,
        call: str,
        *,
        images_png: list[bytes],
        started_at: float,
        response: Any = None,
        text: str | None = None,
        error: Exception | None = None,
    ) -> None:
        """One line per box-detection or orientation call, successful or not.

        On failure it also carries the start and end of the raw reply: these calls return only
        coordinates, edges, angles and confidences, so the excerpt shows why a reply was unusable
        (cut off at the output limit, wrong shape) without holding KYC values.
        """
        usage = getattr(response, "usage_metadata", None)
        input_tokens, text_tokens, thinking_tokens = self._usage_tokens(usage)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        fields = (
            f"{call} call status=%s images=%d image_sizes=%s duration_sec=%.2f finish_reason=%s "
            f"input_tokens=%s output_tokens=%s thinking_tokens=%s reply_chars=%s"
        )
        args = [
            "success" if error is None else f"error:{type(error).__name__}",
            len(images_png), self._image_sizes(images_png), time.perf_counter() - started_at,
            getattr(finish_reason, "name", finish_reason), input_tokens, text_tokens, thinking_tokens,
            len(text) if text is not None else None,
        ]
        if error is None:
            logger.info(fields, *args)
            return
        excerpt = ""
        if text:
            excerpt = text if len(text) <= 400 else f"{text[:300]} ... {text[-100:]}"
        logger.warning(fields + " error=%s reply_excerpt=%r", *args, str(error)[:300], excerpt)

    @staticmethod
    def _usage_tokens(usage: Any) -> tuple[int | None, int | None, int | None]:
        """(prompt, answer text, thinking) for one response.

        Gemini meters thinking separately from the answer rather than inside a single output
        figure, so anything wanting a total has to add the two together.
        """
        return (
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "thoughts_token_count", None),
        )

    @classmethod
    def _output_total(cls, usage: Any) -> int | None:
        _, text_tokens, thinking_tokens = cls._usage_tokens(usage)
        if text_tokens is None and thinking_tokens is None:
            return None
        return (text_tokens or 0) + (thinking_tokens or 0)

    async def extract(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str | None,
        document_name: str,
        document_type: str,
        tenant_type: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        hit_id = uuid.uuid4().hex[:12]
        # A company tenant is read with its own field rules and field list; everything else -
        # security rules, side and document-type observations, output format - is shared.
        if is_company(tenant_type):
            system_instruction, fields = COMPANY_SYSTEM_INSTRUCTION, COMPANY_EXTRACTION_FIELDS
        else:
            system_instruction, fields = SYSTEM_INSTRUCTION, DATA_FIELDS
        system_instruction = system_instruction + "\n" + DOCUMENT_TYPE_CLASSIFICATION_INSTRUCTION
        media_type = content_type or "application/octet-stream"
        if media_type not in {"application/pdf", "image/jpeg", "image/png", "image/webp"}:
            media_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"
        source_type = "document" if media_type == "application/pdf" else "image"
        content, media_type = downscale_image_for_api(content, media_type, self.settings.gemini_max_image_edge)
        # Neither document_type nor document_name is passed to the model - see EXTRACTION_PROMPT.
        # The type stays a parameter because the caller's selection is what the response is later
        # validated against; document_name is a free-typed display string the caller controls (it
        # could just as easily read "Passport" while a national ID is uploaded), so classification
        # must never be able to see it, only the file's own pixels.
        prompt_text = EXTRACTION_PROMPT
        message_content = [
            types.Part.from_bytes(data=content, mime_type=media_type),
            types.Part.from_text(text=prompt_text),
        ]
        try:
            response, prompt_tokens, system_tokens = await self._create_with_token_breakdown(
                prompt_text, message_content, system_instruction,
            )
            log_processing(started_at, "success")
            usage = getattr(response, "usage_metadata", None)
            input_tokens, _, _ = self._usage_tokens(usage)
            record_usage(CALL_EXTRACTION, input_tokens, self._output_total(usage))
            logger.info(
                "Gemini usage input_tokens=%s output_tokens=%s",
                input_tokens if input_tokens is not None else "unknown",
                self._output_total(usage) if self._output_total(usage) is not None else "unknown",
            )
            text = self._response_text(response)
            if not text:
                raise ValueError(f"Gemini returned no text part ({self._finish_details(response)})")
            # Gemini returns plain "Field_Name: value" lines; every structure the caller sees - the
            # data map, missingInfo, errorInfo, success - is built here and in post_processing,
            # never by the model.
            parsed = parse_extraction_text(text, fields)
            input_breakdown = self._build_input_breakdown(source_type, input_tokens, system_tokens, prompt_tokens)
            self._log_hit(hit_id, filename, media_type, started_at, "success", usage, input_breakdown)
            self._log_token_breakdown(hit_id, filename, usage, input_breakdown)
            if self.settings.log_extracted_results:
                self._log_result(hit_id, parsed)
            return parsed
        except Exception as exc:
            log_processing(started_at, f"error:{type(exc).__name__}")
            logger.exception("Gemini OCR request failed")
            self._log_hit(hit_id, filename, media_type, started_at, f"error:{type(exc).__name__}", None)
            raise GeminiServiceError("Gemini OCR request failed") from exc

    async def detect_box_corners(self, page_images_png: list[bytes]) -> list[dict[str, Any]]:
        content: list[Any] = [types.Part.from_bytes(data=png, mime_type="image/png") for png in page_images_png]
        content.append(types.Part.from_text(text=BOX_DETECTION_PROMPT.format(page_count=len(page_images_png))))
        started_at = time.perf_counter()
        response: Any = None
        text: str | None = None
        try:
            response = await self.client.aio.models.generate_content(
                model=self._vision_model,
                contents=content,
                config=self._generation_config(
                    # Headroom for several pages that each contain multiple documents; only tokens
                    # actually generated are billed, so an unused ceiling costs nothing.
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=BOX_DETECTION_SCHEMA,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            input_tokens, _, _ = self._usage_tokens(usage)
            # Recorded before the reply is parsed, so a call whose reply is unusable is still billed
            # in usage.log.
            record_usage(CALL_CROP, input_tokens, self._output_total(usage))
            text = self._response_text(response)
            if not text:
                raise ValueError(f"Gemini returned no text part for box detection ({self._finish_details(response)})")
            parsed = json.loads(text)
            pages = parsed.get("pages")
            if not isinstance(pages, list) or len(pages) != len(page_images_png):
                raise ValueError(f"Expected {len(page_images_png)} page entries in box-detection response, got {pages!r}")
        except Exception as exc:
            self._log_vision_call(
                "Box detection", images_png=page_images_png, started_at=started_at,
                response=response, text=text, error=exc,
            )
            raise
        self._log_vision_call(
            "Box detection", images_png=page_images_png, started_at=started_at, response=response, text=text,
        )
        logger.info(
            "Box detection usage input_tokens=%s output_tokens=%s pages=%d",
            input_tokens, self._output_total(usage), len(pages),
        )
        return pages

    async def segment_documents(self, page_images_png: list[bytes]) -> list[dict[str, Any]]:
        """Which documents a file holds and where - one entry per page, each with its regions.

        Uses GEMINI_MODEL, not the vision model: the type it reports decides what each region is
        checked as, so it is a classification call rather than a simple locating one.
        """
        content: list[Any] = [types.Part.from_bytes(data=png, mime_type="image/png") for png in page_images_png]
        content.append(types.Part.from_text(text=SEGMENTATION_PROMPT.format(page_count=len(page_images_png))))
        started_at = time.perf_counter()
        response: Any = None
        text: str | None = None
        try:
            response = await self.client.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=content,
                config=self._generation_config(
                    # Telling similar documents apart (an Emirates ID from a residence card, an
                    # Ejari from a tenancy contract) is a judgment call, given the same small
                    # thinking budget as verification.
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.settings.gemini_verification_thinking_budget,
                    ),
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=SEGMENTATION_SCHEMA,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            input_tokens, _, _ = self._usage_tokens(usage)
            record_usage(CALL_SEGMENTATION, input_tokens, self._output_total(usage))
            text = self._response_text(response)
            if not text:
                raise ValueError(f"Gemini returned no text part for segmentation ({self._finish_details(response)})")
            pages = json.loads(text).get("pages")
            if not isinstance(pages, list) or len(pages) != len(page_images_png):
                raise ValueError(f"Expected {len(page_images_png)} page entries in segmentation response, got {pages!r}")
        except Exception as exc:
            self._log_vision_call(
                "Segmentation", images_png=page_images_png, started_at=started_at,
                response=response, text=text, error=exc,
            )
            raise
        self._log_vision_call(
            "Segmentation", images_png=page_images_png, started_at=started_at, response=response, text=text,
        )
        return pages

    async def detect_orientations(self, crop_images_png: list[bytes]) -> list[dict[str, Any]]:
        """Judge upright orientation from the already-cropped documents. Reading one isolated
        document is a far cleaner signal than picking orientation out of a full scene, which is
        why this is a separate pass rather than another field on the box-detection call."""
        content: list[Any] = [types.Part.from_bytes(data=png, mime_type="image/png") for png in crop_images_png]
        content.append(types.Part.from_text(text=ORIENTATION_PROMPT.format(document_count=len(crop_images_png))))
        started_at = time.perf_counter()
        response: Any = None
        text: str | None = None
        try:
            response = await self.client.aio.models.generate_content(
                model=self._vision_model,
                contents=content,
                # No sampling controls: Gemini exposes temperature, but pinning it would not make this
                # repeatable on its own. Repeatability comes from constraining the answer instead -
                # top_edge is a four-value enum and the arithmetic lives in code.
                config=self._generation_config(
                    # Thinking tokens count against this same ceiling, and some models think even with
                    # a zero budget (~100-250 tokens seen), so a tight cap cuts off valid replies. With
                    # no free-text field in the schema the reply has nothing to loop on, so a generous
                    # ceiling is safe; only the tokens actually generated are billed.
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=ORIENTATION_SCHEMA,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            input_tokens, _, _ = self._usage_tokens(usage)
            # Recorded before the reply is parsed, so a call whose reply is unusable is still billed
            # in usage.log.
            record_usage(CALL_ROTATION, input_tokens, self._output_total(usage))
            text = self._response_text(response)
            if not text:
                raise ValueError(f"Gemini returned no text part for orientation detection ({self._finish_details(response)})")
            parsed = json.loads(text)
            documents = parsed.get("documents")
            if not isinstance(documents, list) or len(documents) != len(crop_images_png):
                raise ValueError(f"Expected {len(crop_images_png)} orientation entries, got {documents!r}")
        except Exception as exc:
            self._log_vision_call(
                "Orientation", images_png=crop_images_png, started_at=started_at,
                response=response, text=text, error=exc,
            )
            raise
        self._log_vision_call(
            "Orientation", images_png=crop_images_png, started_at=started_at, response=response, text=text,
        )
        logger.info(
            "Orientation usage input_tokens=%s output_tokens=%s documents=%d",
            input_tokens, self._output_total(usage), len(documents),
        )
        return documents

    async def verify_document(
        self, *, content: bytes, media_type: str, document_type: str, today: date,
        short_messages: bool = False,
    ) -> list[str]:
        """Self-verification of one document, done by looking at it again rather than by a Python
        rule engine - completeness, required fields, number formats, date logic, expiry, and
        internal consistency (MRZ against printed text, front against back) are all judgment calls
        Gemini can make directly from the pixels, including for fields this service never captures
        into its own extracted `data` (a cheque's amount, a trade license's activities). The result
        is nothing more than a list of problem sentences - empty when the document is fine - so the
        caller has only to turn each one into an existing errorInfo entry, never a new field.
        """
        content, media_type = downscale_image_for_api(content, media_type, self.settings.gemini_max_image_edge)
        # Spelled out in words as well as digits, and in the day/month/year order the documents
        # themselves use rather than ISO - a bare "2026-09-23" invites exactly the kind of
        # misreading (year-first, or confused for DD-MM-YY) that turns a document's own genuinely
        # past date into a false "this is in the future" finding.
        today_text = f"{today.strftime('%d/%m/%Y')} ({today.strftime('%d %B %Y')})"
        prompt = SELF_VERIFICATION_PROMPT.format(document_type=document_type, today=today_text)
        if short_messages:
            prompt += SHORT_SELF_MESSAGES
        message_content = [
            types.Part.from_bytes(data=content, mime_type=media_type),
            types.Part.from_text(text=prompt),
        ]
        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=message_content,
            config=self._generation_config(
                # Unlike extraction, this is a judgment call, not a direct transcription - date
                # arithmetic in particular (is this expiry before or after today) is exactly the
                # kind of thing a model gets wrong more often when forced to answer in one shot with
                # no room to work through it. A small non-zero budget overrides the shared default.
                thinking_config=types.ThinkingConfig(thinking_budget=self.settings.gemini_verification_thinking_budget),
                max_output_tokens=1536,
                response_mime_type="application/json",
                response_schema=ISSUES_SCHEMA,
            ),
        )
        text = self._response_text(response)
        if not text:
            raise ValueError(f"Gemini returned no text part for self-verification ({self._finish_details(response)})")
        parsed = json.loads(text)
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            raise ValueError(f"Expected an issues list from self-verification, got {issues!r}")
        usage = getattr(response, "usage_metadata", None)
        input_tokens, _, _ = self._usage_tokens(usage)
        record_usage(CALL_VALIDATION, input_tokens, self._output_total(usage))
        logger.info(
            "Self-verification usage input_tokens=%s output_tokens=%s issues=%d",
            input_tokens, self._output_total(usage), len(issues),
        )
        return [str(issue) for issue in issues]

    async def verify_documents_cross(
        self, *, documents: list[dict[str, Any]], tenant_type: str | None = None,
        short_messages: bool = False,
    ) -> list[str]:
        """Cross-document verification over already-extracted fields, text-only.

        `documents` is `{"label", "document_type", "data"}` per document - the same fields the
        caller's own response already carries, nothing re-sent from the original images. Comparing
        the extracted text is enough to catch a disagreement between documents (this is what a
        Python field-by-field comparison would otherwise do), and doing it as one more Gemini call
        rather than a second rule engine means the reasoning about which fields are the "same fact"
        across two different document types - a bank statement's account holder against a National
        ID's holder - is the model's own judgment, not a hardcoded mapping this service has to keep
        in step with every new document type.
        """
        def _describe(document: dict[str, Any]) -> str:
            lines = [f"{document['label']} ({document['document_type']}):"]
            fields = [f"  {field}: {value}" for field, value in document["data"].items() if value is not None]
            lines.extend(fields or ["  (no fields were read from this document)"])
            return "\n".join(lines)

        template = COMPANY_CROSS_VERIFICATION_PROMPT if is_company(tenant_type) else CROSS_VERIFICATION_PROMPT
        prompt = template.format(
            count=len(documents),
            documents_text="\n\n".join(_describe(document) for document in documents),
        )
        if short_messages:
            prompt += SHORT_CROSS_MESSAGES
        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=[types.Part.from_text(text=prompt)],
            config=self._generation_config(
                # Same reasoning as self-verification: comparing values across documents is a
                # judgment call, not direct transcription, and benefits from the same small
                # thinking budget instead of the shared zero default.
                thinking_config=types.ThinkingConfig(thinking_budget=self.settings.gemini_verification_thinking_budget),
                max_output_tokens=1536,
                response_mime_type="application/json",
                response_schema=ISSUES_SCHEMA,
            ),
        )
        text = self._response_text(response)
        if not text:
            raise ValueError(f"Gemini returned no text part for cross-verification ({self._finish_details(response)})")
        parsed = json.loads(text)
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            raise ValueError(f"Expected an issues list from cross-verification, got {issues!r}")
        usage = getattr(response, "usage_metadata", None)
        input_tokens, _, _ = self._usage_tokens(usage)
        record_usage(CALL_VALIDATION, input_tokens, self._output_total(usage))
        logger.info(
            "Cross-verification usage input_tokens=%s output_tokens=%s documents=%d issues=%d",
            input_tokens, self._output_total(usage), len(documents), len(issues),
        )
        return [str(issue) for issue in issues]

    async def verify_document_pair(
        self, *, first_content: bytes, first_mime: str, second_content: bytes, second_mime: str, document_type: str,
    ) -> bool:
        """Confirms two separately-uploaded, individually-incomplete files are the front and back
        of the same document before they are merged into one. A judgment call made by looking at
        both images together, not a hardcoded rule - the same reasoning self- and cross-verification
        already use - so it can catch two different people's split uploads landing in the same
        batch, rather than merging them on the strength of matching document types alone.
        """
        first_content, first_mime = downscale_image_for_api(
            first_content, first_mime, self.settings.gemini_max_image_edge
        )
        second_content, second_mime = downscale_image_for_api(
            second_content, second_mime, self.settings.gemini_max_image_edge
        )
        prompt = PAIR_CONFIRMATION_PROMPT.format(document_type=document_type)
        message_content = [
            types.Part.from_bytes(data=first_content, mime_type=first_mime),
            types.Part.from_bytes(data=second_content, mime_type=second_mime),
            types.Part.from_text(text=prompt),
        ]
        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=message_content,
            config=self._generation_config(
                # Same reasoning as self- and cross-verification: this is a judgment call, not a
                # direct transcription, and benefits from a small non-zero thinking budget.
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                max_output_tokens=512,
                response_mime_type="application/json",
                response_schema=PAIR_CONFIRMATION_SCHEMA,
            ),
        )
        text = self._response_text(response)
        if not text:
            raise ValueError(f"Gemini returned no text part for pair confirmation ({self._finish_details(response)})")
        parsed = json.loads(text)
        same_person = parsed.get("samePerson")
        if not isinstance(same_person, bool):
            raise ValueError(f"Expected a samePerson boolean from pair confirmation, got {same_person!r}")
        usage = getattr(response, "usage_metadata", None)
        input_tokens, _, _ = self._usage_tokens(usage)
        record_usage(CALL_VALIDATION, input_tokens, self._output_total(usage))
        logger.info(
            "Pair confirmation usage input_tokens=%s output_tokens=%s same_person=%s",
            input_tokens, self._output_total(usage), same_person,
        )
        return same_person

    async def country_code(self, *, value: str, kind: str) -> str | None:
        """The ISO 3166-1 alpha-2 code a country code, name or nationality refers to, or None.

        Text-only and tiny: used once per value the curated country lists do not know, after which
        the answer is stored and reused - see app.services.country_learning.
        """
        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=[types.Part.from_text(text=COUNTRY_CODE_PROMPT.format(kind=kind, value=value))],
            config=self._generation_config(
                # Only a cap - billing is for the tokens actually used. The answer itself is a few
                # tokens, but newer models still think first - a few hundred tokens on a hard value
                # such as "Kosovar" - and a tight cap cuts the answer off.
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=COUNTRY_CODE_SCHEMA,
            ),
        )
        text = self._response_text(response)
        if not text:
            raise ValueError(f"Gemini returned no text part for country matching ({self._finish_details(response)})")
        code = str(json.loads(text).get("code") or "").strip().upper()
        usage = getattr(response, "usage_metadata", None)
        input_tokens, _, _ = self._usage_tokens(usage)
        logger.info(
            "Country code usage input_tokens=%s output_tokens=%s kind=%s code=%s",
            input_tokens, self._output_total(usage), kind, code,
        )
        return code if len(code) == 2 and code.isalpha() else None

    async def _create_with_token_breakdown(
        self, prompt_text: str, message_content: list[Any], system_instruction: str,
    ) -> tuple[Any, int | None, int | None]:
        async def count_prompt_tokens() -> int | None:
            try:
                result = await self.client.aio.models.count_tokens(
                    model=self.settings.gemini_model, contents=prompt_text
                )
                return result.total_tokens
            except Exception:
                logger.warning("Could not measure prompt token count", exc_info=True)
                return None

        # Run the real call alongside the (cheap, text-only) token-counting calls so the breakdown
        # adds no serial latency to the OCR request itself.
        response, prompt_tokens, system_tokens = await asyncio.gather(
            self.client.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=message_content,
                config=self._generation_config(
                    max_output_tokens=self.settings.gemini_max_tokens,
                    system_instruction=system_instruction,
                ),
            ),
            count_prompt_tokens(),
            self._get_system_tokens(system_instruction),
        )
        return response, prompt_tokens, system_tokens

    def _build_input_breakdown(
        self,
        source_type: str,
        input_tokens: int | None,
        system_tokens: int | None,
        prompt_tokens: int | None,
    ) -> dict[str, int | None]:
        # The image/PDF bytes aren't counted directly (that would mean re-sending the whole file
        # through count_tokens just to measure it); they're the residual once the independently
        # measured system and prompt tokens come off the billed prompt_token_count.
        document_tokens = None
        if input_tokens is not None and system_tokens is not None and prompt_tokens is not None:
            document_tokens = input_tokens - system_tokens - prompt_tokens
        is_pdf = source_type == "document"
        return {
            "system_instruction_tokens": system_tokens,
            "prompt_tokens": prompt_tokens,
            "image_tokens": document_tokens if not is_pdf else None,
            "pdf_tokens": document_tokens if is_pdf else None,
        }

    def _log_hit(
        self,
        hit_id: str,
        filename: str,
        content_type: str,
        started_at: float,
        status: str,
        usage: Any,
        input_breakdown: dict[str, int | None] | None = None,
    ) -> None:
        input_tokens, _, _ = self._usage_tokens(usage)
        output_tokens = self._output_total(usage)
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        hits_logger.info(json.dumps({
            "hit_id": hit_id,
            **log_timestamps(),
            "filename": filename,
            "content_type": content_type,
            "model": self.settings.gemini_model,
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": getattr(usage, "cached_content_token_count", None),
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            "estimated_cost_usd": None,
            "input_token_breakdown": input_breakdown,
        }))

    def _log_result(self, hit_id: str, result: dict[str, Any]) -> None:
        results_logger.info(json.dumps({"hit_id": hit_id, **log_timestamps(), "result": result}))

    def _log_token_breakdown(
        self,
        hit_id: str,
        filename: str,
        usage: Any,
        input_breakdown: dict[str, int | None],
    ) -> None:
        input_tokens, text_output_tokens, thinking_tokens = self._usage_tokens(usage)
        token_breakdown_logger.info(json.dumps({
            "hit_id": hit_id,
            **log_timestamps(),
            "filename": filename,
            **input_breakdown,
            "thinking_output_tokens": thinking_tokens,
            "text_output_tokens": text_output_tokens,
            "input_tokens_total": input_tokens,
            "output_tokens_total": self._output_total(usage),
        }))


@lru_cache
def get_gemini_service() -> GeminiService:
    """The one shared client for the whole process - every request uses the same instance."""
    return GeminiService(get_settings())
