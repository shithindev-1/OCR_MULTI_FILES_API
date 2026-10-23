"""Front/back pairing for one request: the pipeline's own pairing run per document type, and -
when one request may hold several people's cards - matching several fronts to several backs."""

import logging
import re
from functools import partial
from typing import Any

from app.core.config import settings
from app.core.exceptions import ValidationStopped
from app.services.concurrency import run_concurrently
from app.services.gemini import charged_to, documents_named, get_gemini_service
from app.services.leasing.reports import first_error_message, indices_named
from app.services.ocr_pipeline import (
    VerificationUnavailable,
    call_with_one_retry,
    resolve_incomplete_pairs,
)

logger = logging.getLogger("uae_ocr.leasing")


_SIDE_ID_FIELDS = ("National_Id", "Signatory_National_Id")


def _side_id(candidate: dict[str, Any]) -> str | None:
    """The ID number read off one side of a card, digits and letters only."""
    for field in _SIDE_ID_FIELDS:
        value = re.sub(r"[^A-Za-z0-9]", "", str(candidate["data"].get(field) or "")).upper()
        if len(value) >= 5:
            return value
    return None


async def match_sides(group: list[dict[str, Any]], *, request_id: str) -> list[list[dict[str, Any]]]:
    """Splits one type's uploads into groups the pipeline's pairing can resolve - for a request that
    may hold several people's cards, such as a husband's and a wife's ID fronts and backs.

    Each front is matched to its back, first by the ID number both sides print (no model call),
    then, for those left, by asking the model about every remaining front/back combination. A match
    is kept only when it is one-to-one: that front fits only that back, and that back only that
    front. Each match becomes its own [front, back] group; everything else stays in one group, where
    the pipeline reports a lone side or an ambiguous set exactly as before.
    """
    fronts = [c for c in group if not c["is_complete"] and c["front_visible"] and not c["back_visible"]]
    backs = [c for c in group if not c["is_complete"] and c["back_visible"] and not c["front_visible"]]
    if len(fronts) < 2 and len(backs) < 2:
        return [group]

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for front in list(fronts):
        number = _side_id(front)
        same = [back for back in backs if number and _side_id(back) == number]
        if len(same) == 1 and sum(_side_id(other) == number for other in fronts) == 1:
            pairs.append((front, same[0]))
            fronts.remove(front)
            backs.remove(same[0])

    if fronts and backs:
        if settings.validation_enabled:
            @charged_to(lambda front, back: documents_named(front["document_name"], front["document_type"])
                        + documents_named(back["document_name"], back["document_type"]))
            async def confirm(front: dict[str, Any], back: dict[str, Any]) -> bool:
                try:
                    return await call_with_one_retry(
                        lambda: get_gemini_service().verify_document_pair(
                            first_content=front["content"], first_mime=front["mime_type"],
                            second_content=back["content"], second_mime=back["mime_type"],
                            document_type=front["document_type"],
                        ),
                        what="Pair confirmation", request_id=request_id,
                    )
                except VerificationUnavailable:
                    return False

            combinations = [(front, back) for front in fronts for back in backs]
            answers = await run_concurrently([partial(confirm, front, back) for front, back in combinations])
            fits = [(front, back) for (front, back), same in zip(combinations, answers) if same]
            for front, back in fits:
                if sum(f is front for f, _ in fits) == 1 and sum(b is back for _, b in fits) == 1:
                    front["_pair_confirmed_with"] = back["sha256"]
                    pairs.append((front, back))
        elif len(fronts) == 1 and len(backs) == 1:
            pairs.append((fronts[0], backs[0]))

    paired = {id(side) for pair in pairs for side in pair}
    logger.info(
        "Front/back matching request_id=%s document_type=%s uploads=%d pairs=%d",
        request_id, group[0]["document_type"], len(group), len(pairs),
    )
    rest = [candidate for candidate in group if id(candidate) not in paired]
    return [list(pair) for pair in pairs] + ([rest] if rest else [])


async def resolve_pairs_per_type(
    *, request_id: str, candidates: list[dict[str, Any]], index_by_name: dict[str, int],
    documents_report: list[dict[str, Any]], started_at: float, match_several: bool = False,
) -> list[dict[str, Any]]:
    """Runs the pipeline's own front/back pairing once per document type, so a problem with one type's
    uploads is reported on those documents only and never stops another type from being paired.
    Returns every document that resolved cleanly (complete originals plus merged pairs).

    `match_several` (type-detection endpoint, where one request may hold several people's cards)
    first matches several fronts to several backs - see `match_sides`."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate["document_type"], []).append(candidate)

    groups: list[list[dict[str, Any]]] = []
    for group in by_type.values():
        groups.extend(await match_sides(group, request_id=request_id) if match_several else [group])

    resolved: list[dict[str, Any]] = []
    for group in groups:
        try:
            resolved.extend(await resolve_incomplete_pairs(
                request_id=request_id,
                candidates=group,
                validation_enabled=settings.validation_enabled,
                started_at=started_at,
            ))
        except ValidationStopped as stopped:
            message = first_error_message(stopped.response)
            errors = stopped.response.get("errorInfo") or [{}]
            incomplete = [c for c in group if not c["is_complete"]]
            named = indices_named(
                errors[0].get("DocumentName") or "", index_by_name,
                [index_by_name[c["document_name"]] for c in incomplete],
            )
            for index in named:
                documents_report[index]["error"] = message
            # Complete documents in this group were never the problem - they still go on to
            # self-verification. An incomplete one the error did not name (e.g. a front whose only
            # possible partner was the unpairable upload) is still incomplete: pair it alone, which
            # returns its own lone-incomplete error with no Gemini call.
            resolved.extend(c for c in group if c["is_complete"])
            for candidate in incomplete:
                index = index_by_name[candidate["document_name"]]
                if index in named:
                    continue
                try:
                    await resolve_incomplete_pairs(
                        request_id=request_id, candidates=[candidate],
                        validation_enabled=settings.validation_enabled, started_at=started_at,
                    )
                except ValidationStopped as lone:
                    documents_report[index]["error"] = first_error_message(lone.response)
    return resolved
