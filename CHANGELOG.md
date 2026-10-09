# Changelog

All notable changes to this API. Versions follow [semantic versioning](https://semver.org/): a new
major version means a change that existing callers must adapt to.

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
    type: `National_ID.pdf`, `Passport.jpg`, `National_ID_2.pdf` - or, with several tenants,
    `Tenant_1_National_ID.pdf`, `Tenant_2_Passport.jpg`. An original is never overwritten.
- **Logs:** segmentation, front/back matching, tenant grouping and per-tenant results in
  `ocr-api.log`; the per-request usage line has a `segmentation` figure.

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
