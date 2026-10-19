# Changelog

All notable changes to this API. Versions follow [semantic versioning](https://semver.org/): a new
major version means a change that existing callers must adapt to.

## Unreleased

### Fixed (type-detection endpoints)
- **Small cards on a large scan were taken as the whole page.** An ID card on an A4 scan is under 4% of
  the page, below the 5% sanity limit, so its region became the whole page - which then painted every
  other document grey (a passport and visa "fully covered", the ID PDF the whole page twice). Split
  regions now only need 0.5% of the page, and a region that falls back to the whole page paints nothing.
- **A document that doesn't belong to the tenant is now rejected on its own.** Every supporting
  document must be shown to belong to a tenant - also when there is only one: an individual's
  document must print their name or ID number, and a trade licence or initial approval must LIST
  them; a company's document must name the company or its licence number. One that doesn't gets an
  error (e.g. "'Trade licence' doesn't list the tenant…"), no processed file and no data in
  `form_data` - the tenant's own valid documents are still extracted and returned.
- A trade licence that lists an individual is accepted for them, but adds nothing to their
  `form_data`.
- **False cross-check failures.** A passport's home-country national number is no longer taken as
  the tenant's National ID; occupation and employer are no longer compared across documents; and
  the check's answer is no longer cut off (which showed as "could not be checked right now").
  `form_data` takes a value from the National ID first, then the visa, then the passport.

### Changed (all endpoints - tell the .NET team)
- **Dates in `form_data` are now `DD-MM-YYYY`** (e.g. `14-10-2026`, the leasing form's format) instead of
  `DD/MM/YYYY`, on the old and the new endpoints. Every printed format is still recognised
  (`04-Apr-2024`, `2024/04/28`, `28/04/2024`, `15-04-2026` -> `04-04-2024`, `28-04-2024`, `28-04-2024`, `15-04-2026`).

### Maintenance
- **`usage.log` now counts the split (segmentation) call of a type-detection request** - its tokens were
  recorded and then wiped when the batch started its own tally, so detect totals were ~2,000 tokens per
  photo too low. A detect request rejected after the split (no National ID / trade licence, or an
  unreadable file) now also writes its usage line.
- Decoding settings in `.env` for every Gemini call: `GEMINI_TEMPERATURE=0`, `GEMINI_SEED=0`,
  `GEMINI_TOP_P=1`, `GEMINI_TOP_K=1` (always the single most likely answer).
- `install-service.ps1` takes `-DisplayName` (default: "UAE OCR API" for UaeOcrApi, else the service
  name) and refuses a display name another service already uses.
- A rejected background job logs its plain-English reason (`reason=...`), not only the status code.
- The system-instruction token count, which the Gemini Developer API does not support, is tried once per
  instruction and logged once at INFO, instead of a warning with a traceback on every extraction.
- The tests run without a local `.env` (conftest sets the required settings and plain HTTP).

### Changed (type-detection endpoints)
- **Messages say where a document was found** when its file holds more than one document, e.g. "The visa
  found on page 4 of 'Trade License 11.pdf' expired on 28-04-2024..." instead of "Your visa expired...". A file
  holding a single document keeps the short message.
- **Overlapping documents in one photo:** the split call also reports which documents lie on top of each
  region (`covered_by`); each cut-out is taken from the document's exact outline with no margin into a
  neighbour, and any part another document lies on is blanked light grey - including the other side of the
  same card lying on top (e.g. an ID's front over its back). A covered part only fails a document when it
  hides a required detail. Two different documents are never merged as duplicates, however much they overlap.
- **Cross-check messages are written per document:** when documents disagree (e.g. a visa and a passport
  with different passport numbers), each document gets its own message from its own point of view instead of
  one shared sentence. The model now reports what disagrees (documents + detail); the wording is done in code.
- Dates inside `documents[].error` messages are written as `DD-MM-YYYY` too.
- **Company: a trade licence or an initial approval** now satisfies the mandatory-document rule
  (the old `/ocr/leasing` endpoints still require a trade licence). An initial approval anchors the
  company like a licence; a licence and an approval of the same company are one tenant.
- **New company `form_data` keys** (type-detection endpoints only): `initial_approval_number`,
  `initial_approval_issue_date`, `initial_approval_expiry`, `initial_approval_issuer` - read only from
  an initial approval. The `trade_license_*` keys are read only from a trade licence.
- **Any trade licence layout** (mainland or free zone; commercial, professional, industrial, business,
  e-licence) is recognised as a trade licence.
- **Processed file names always start with the tenant**, also when there is a single tenant:
  `Tenant_1_National_ID.pdf`, `Tenant_1_Passport.pdf` (was `National_ID.pdf`, `Passport.pdf`).

## 2.1.0 - 2026-10-09

The existing endpoints keep their behaviour and response shapes.

### Added
- **Type-detection endpoints**, `POST /api/v1/ocr/leasing/detect` (JSON `sources[{source}]`) and
  `/detect/upload` (form-data `file`, or `sources[n][source]`), plus background jobs
  `/detect/jobs` and `/detect/jobs/upload`, polled with the normal
  `GET /api/v1/ocr/leasing/jobs/{job_id}`. Only `tenant_type` and `document_refnumber` are sent with
  the files; `document_type`, `document_name` and `document_id` are rejected.
  - Each file is split into the documents it holds; a document's type is accepted only when two
    independent readings agree. File names are never used. Unsupported documents are `ignored`.
  - Several tenants per request: documents are grouped by person (shared ID number, or matching
    name and date of birth); for a company, one tenant per trade licence and each listed person
    uploaded. Several people's ID fronts and backs sent as separate files are matched by ID number,
    then by a same-card check.
  - Each tenant needs its own National ID (individual) or a signatory listed on its trade licence
    (company), and is judged on its own. A request with no National ID
    (individual) or trade licence (company) at all is rejected with the usual message.
  - Short, plain-English error messages; documents are named by their type ("Passport",
    "National ID 2").
  - Response: `form_data` is an array of `{tenant_id, status, ...fields}`, `missing` an array of
    `{tenant_id, fields}`; each document has `tenant_ids`, `document_type`, `status`
    (`valid` / `invalid` / `ignored` / `not_checked`), `merged`, `sources` and `error`.
  - Processed (cropped/rotated/merged) copies are uploaded beside the original, named after their
    type, always as a PDF: `National_ID.pdf`, `Passport.pdf`, `National_ID_2.pdf` - or, with several
    tenants, `Tenant_1_National_ID.pdf`, `Tenant_2_Passport.pdf`. An original is never overwritten.
    Each is also kept on the server under `LEASING_OUTPUT_DIR/<document_refnumber>/`; an attached file
    with no blob link gets that local path as its `ocr_file`.
- **Logs:** segmentation, front/back matching, tenant grouping and per-tenant results in
  `ocr-api.log`; the per-request usage line has a `segmentation` figure.

### Fixed
- **Several documents in one photo** (e.g. passport, ID front and back, and residence card) came out
  as one combined file. The split call now asks for boxes in Gemini's own `[ymin, xmin, ymax, xmax]`
  0-1000 format, which it answers in consistently; corner answers on a 0-1000 scale are also read
  correctly. Each document now gets its own PDF.
- **Blurry processed files.** Cut-outs are now lossless (PNG, or a near-lossless PDF read back at
  its own pixel size instead of shrunk to two thirds), crops use cubic warps and Lanczos upscaling,
  the contrast boost is off by default (`DETECTION_CROP_CONTRAST_CLIP=0`), sharpening is lighter
  (`DETECTION_CROP_SHARPEN_AMOUNT=0.4`), and PDF pages are stored at JPEG quality 95 with full colour
  detail. Small print in the processed files is now as sharp as in the upload.

### Changed
- **`usage.log`** has a new `segmentation` key on every request (`0` on the existing endpoints).
  Anything that parses that file must allow it.
- **Code structure:** `services/gemini`, `services/leasing` and `services/type_detection` are now
  packages of focused modules; shared helpers live in `services/concurrency.py`, `services/names.py`
  and `api/forms.py`. No URL, request or response changed.
- **Tooling:** `ruff` (lint and import order) added to `requirements-dev.txt` and configured in
  `pyproject.toml`.

## 2.0.0 - 2026-10-07

### Breaking changes - callers must update
- **No top-level `error` in leasing responses.** Read `status` (`success` / `failed`) and each
  document's own `documents[].error`.
- **422 responses carry one plain sentence** in `detail` (naming the missing fields) instead of a
  list of validation objects.
- **Empty fields are rejected up front.** An empty `source`, `document_id`, `document_type` or
  `document_name` now answers `422` immediately instead of a `failed` document in the result.
- **`tenant_type: company` requires a trade licence** (`document_type` `trade_license`), otherwise
  the request answers `400`.
- **New `form_data` keys** for individuals (see Added). Code that expects exactly the old 15 keys
  must allow the new ones.

### Added
- **Company tenants** (`tenant_type: company`): company name, trade licence details and the
  signatory's ID, passport and visa details, with their own extraction and cross-verification rules.
  The signatory must be listed on the trade licence.
- **Individual fields:** home and residential address (country, state/emirate, city, area, street,
  building, door, P.O. Box) and the passport's issue date.
- **Documents from any country:** national ID cards, passports and visas/residence permits.
- **Progress-bar jobs:** `POST /api/v1/ocr/leasing/jobs` (JSON) and `/jobs/upload` (form-data)
  return a `job_id` at once; `GET /api/v1/ocr/leasing/jobs/{job_id}` returns `jobId`, `percentage`,
  `status` (`Processing` / `Completed`) and, once completed, the normal result.
- **Result reuse** for byte-identical files (`RESULT_CACHE_SECONDS`, memory only).
- **Detailed timing logs** per stage, per download and per vision call.
- New settings: `RESULT_CACHE_SECONDS`, `GEMINI_VERIFICATION_THINKING_BUDGET`,
  `GEMINI_VISION_MODEL`, `LOG_DIR`.

### Changed
- **Faster:** download, extraction, self-verification and crop/upload run for all documents in
  parallel; a merged front/back National ID is no longer read a second time; connections are
  opened at startup.
- **Clearer errors:** every message is short, plain English - the reason first, then what to do.
- **Fewer false rejections:** blank optional fields (accompanied by, sponsor, U.I.D. when an ID
  number is shown, ...) no longer fail a document.
- **Safer verification:** a verification call that fails is retried once, then fails the document
  instead of passing it unchecked.
- Duplicate crop regions are dropped only when two boxes are nearly identical, so an ID's front and
  back are never merged into one crop.

## 1.0.0 - 2026-09-28

- First release: `/api/v1/ocr` and the leasing endpoints (`/api/v1/ocr/leasing`, `/upload`),
  `X-API-Key` authentication, HTTPS enforcement and rate limiting, Azure Blob Storage sources and
  results, crop/rotate pipeline, Windows service deployment with NSSM.
