# Document OCR API

A FastAPI service that extracts identity and leasing-document data - national ID cards, passports and visas from any country, trade licences, Ejari certificates and more - with Gemini's vision API, verifies them, then crops, de-skews and rotates each document.

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | Liveness probe |
| `POST /api/v1/ocr` | `X-API-Key` | Multipart OCR, one merged response |
| `POST /api/v1/ocr/leasing` | `X-API-Key` | Leasing OCR, documents given as URLs (JSON) |
| `POST /api/v1/ocr/leasing/upload` | `X-API-Key` | Leasing OCR, documents attached as files (form-data) |
| `POST /api/v1/ocr/leasing/jobs` | `X-API-Key` | Same as `/ocr/leasing`, run in the background - returns a `job_id` at once |
| `POST /api/v1/ocr/leasing/jobs/upload` | `X-API-Key` | Same as `/ocr/leasing/upload`, run in the background - returns a `job_id` |
| `GET /api/v1/ocr/leasing/jobs/{job_id}` | `X-API-Key` | Progress (`percentage`, `status`) and, once `Completed`, the full `result` |
| `GET /api/v1/ocr/leasing/files?url=...` | `X-API-Key` | View an uploaded `<name>_ocr.<ext>` result |
| `POST /api/v1/ocr/leasing/detect` | `X-API-Key` | Type detection: documents given as URLs (JSON), no type/name/id - [see below](#type-detection-endpoints) |
| `POST /api/v1/ocr/leasing/detect/upload` | `X-API-Key` | Type detection: documents attached as files (form-data) |
| `POST /api/v1/ocr/leasing/detect/jobs` | `X-API-Key` | Same as `/detect`, in the background - poll `GET /ocr/leasing/jobs/{job_id}` |
| `POST /api/v1/ocr/leasing/detect/jobs/upload` | `X-API-Key` | Same as `/detect/upload`, in the background |
| `GET /docs` | none | Swagger UI (disable with `DOCS_ENABLED=false`) |

`X-API-Key` is enforced whenever `API_KEYS` is set; leave it empty only for local testing.

Step-by-step Postman testing: [docs/POSTMAN_GUIDE.md](docs/POSTMAN_GUIDE.md). Release notes: [CHANGELOG.md](CHANGELOG.md).

## Setup

1. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-dev.txt
   ```

   `requirements.txt` holds the runtime dependencies only; `requirements-dev.txt` adds the test tools. Always go through `python -m ...` (`python -m pip`, `python -m pytest`, `python -m app`): Windows Application Control / Smart App Control can block the unsigned `pip.exe` and `uvicorn.exe` launchers inside `.venv\Scripts`, while the signed `python.exe` runs fine.

2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` and `GEMINI_MODEL`. For local HTTP-only development, set `REQUIRE_HTTPS=false` in `.env`.

3. Start the API:

   ```powershell
   python -m uvicorn app.main:app --reload   # development, auto-reload
   python -m app                             # production mode, host/port from APP_HOST/APP_PORT
   ```

   Then open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

The production API must run behind HTTPS/TLS. With `REQUIRE_HTTPS=true` (the default), `/ocr` rejects plain HTTP even if a client reaches the process directly. Terminate TLS at a reverse proxy such as IIS, nginx, or Caddy, forward `X-Forwarded-Proto: https`, and firewall the Uvicorn port so it is not internet-facing. Plain HTTP is only for local development with `REQUIRE_HTTPS=false`.

API keys and uploaded document bytes are never written to logs. `logs/ocr-api.log` and `logs/errors.log` contain only request ID, document type, filename, size, MIME type, content hash, processing time, status, and safe error type — never extracted KYC values.

`logs/rotation.log` records one JSON line per cropped document: the rotation angle Gemini asked for, its confidence and reasoning, what OpenCV actually applied, and where on the page the crop came from — `crop_box` (the four corners in the source image's own pixels, ordered top-left, top-right, bottom-right, bottom-left, as `crop_page` consumes them), `crop_bbox` (the axis-aligned bounds) and `page_size`. No extracted values.

`logs/ocr-api.log` contains every request-lifecycle entry. `logs/errors.log` duplicates only the WARNING/ERROR entries (rejections, rate-limit hits, Gemini failures, unhandled exceptions) for fast incident triage. `logs/hits.log` records one JSON line per Gemini call (`hit_id`, timestamp, filename, model, status, token counts, duration) — no extracted values. `logs/results.log` is **off by default**; `LOG_EXTRACTED_RESULTS=true` makes it record the full parsed extraction result per `hit_id` — development only, see the retention policy below. `logs/token-breakdown.log` records one JSON line per successful Gemini call splitting tokens into `image_input_tokens`/`image_output_tokens` or `pdf_input_tokens`/`pdf_output_tokens` (whichever content type that hit sent) plus `thinking_output_tokens`; Google's usage API reports one combined `input_tokens` figure per request and does not meter the image/PDF bytes separately from the prompt text, and thinking has no input-side token count, so the inapplicable fields are always `null`. All five files rotate daily and are pruned after `LOG_RETENTION_DAYS`.

Every log line is stamped in three zones. The JSON logs carry `timestamp` (UTC, the canonical one), `timestamp_uae` (UTC+4) and `timestamp_ist` (UTC+5:30) as separate fields; the plain-text logs render all three into their single time column. The offsets are fixed rather than looked up through `zoneinfo`, since neither zone observes DST — so the offsets are exact year-round — and Windows has no IANA database without the extra `tzdata` package. The two local stamps keep their own dates, because past 20:00 UTC the UAE and Indian dates have already rolled over.

## Project structure

```
app/
  __main__.py               `python -m app`: production entry point (Uvicorn, one worker)
  main.py                   create_app(): middleware, CORS, exception handlers, routers
  core/
    config.py               Settings (environment variables / .env)
    logging.py              rotating log files, UTC/UAE/IST timestamps
    exceptions.py           ValidationStopped + exception handlers
    middleware.py           HTTPS enforcement and rate limiting for the OCR endpoints
    security.py             X-API-Key authentication for /api/v1
  api/
    router.py               /health at the root; everything else under /api/v1 behind X-API-Key
    forms.py                form-data helpers shared by the leasing routes
    routes/
      health.py             GET  /health
      ocr.py                POST /api/v1/ocr
      leasing.py            /api/v1/ocr/leasing (+ /upload), background jobs, result files
      leasing_detect.py     /api/v1/ocr/leasing/detect (+ /upload, /jobs, /jobs/upload)
  schemas/
    ocr.py                  field lists (individual + company), document types, response models
    leasing.py              /ocr/leasing request and response models
    detect.py               /ocr/leasing/detect request and response models
  services/
    gemini/                 everything sent to Gemini
      client.py             GeminiService - every model call - and the shared instance
      prompts.py            the instructions, prompts and JSON response schemas
      usage.py              per-request token accounting (usage.log)
    leasing/                /ocr/leasing batch processing
      batch.py              process_batch / process_sources: the stages, in order
      rules.py              request checks made before anything is downloaded
      sources.py            downloading from an allowed URL or the blob container
      pairing.py            front/back pairing, per document type (several people's cards too)
      reports.py            the response and its per-document entries
      outputs.py            output folders; the processed file uploaded beside each original
      signatory.py          a company signatory's role from the trade licence
      tenant_results.py     several tenants in one request: judged and filled one by one
    type_detection/         the /ocr/leasing/detect endpoints
      service.py            runs a request: split files, run the leasing batch, build the response
      segmentation.py       splits one file into the documents it holds
      tenants.py            groups documents into tenants (people, or companies and signatories)
      responses.py          the response shape: form_data and missing per tenant
    ocr_pipeline.py         the shared pipeline: extract, pair/merge, verify, crop
    detection.py            OpenCV page rendering, crop, rotation, enhancement
    concurrency.py          running one stage's documents in parallel, with a cap
    names.py                tolerant comparison of names read off different documents
    jobs.py                 background jobs for the progress-bar endpoints
    progress.py             overall completion percentage across all documents of a job
    result_cache.py         in-memory reuse of results for byte-identical files
    retention.py            hourly clean-up of stored originals and crops
    post_processing.py      parse model output, build /ocr error payloads
    form_normalization.py   form_data formatting: gender, DD-MM-YYYY dates, country names
    countries.py            country lookup over app/data (codes, nationalities)
    country_learning.py     new country codes/nationalities: saved, matched by Gemini in the background
    validation.py           upload validation and rate limiting
    blob_storage.py         Azure Blob Storage fetch/upload for /ocr/leasing
  data/
    country_codes.json      code -> exact country name (2-letter, 3-letter, UAE/UK/KSA...)
    nationalities.json      nationality or other name -> exact country name ("Syrian" -> "Syria")
deploy/windows/             NSSM install / update / uninstall scripts
docs/POSTMAN_GUIDE.md       step-by-step Postman testing
tests/                      offline tests - Gemini is always stubbed
```

Run the tests with `python -m pytest -q`. They make no network calls. Lint with `python -m ruff check app tests` (configured in `pyproject.toml`).

## Leasing endpoints at a glance

**Tenant types.** `tenant_type` decides which fields are extracted into `form_data`:

| `tenant_type` | `form_data` fields | Required document |
| --- | --- | --- |
| `individual` (default) | tenant name (EN/AR), date of birth, nationality, gender, occupation, employer, national ID + expiry, passport number + issuing place + issue date + expiry, visa number + start + expiry, home and residential address (country, state/emirate, city, area, street, building, door, P.O. Box) | `national_id` - an individual request without one is rejected at once |
| `company` | company name (EN/AR), trade licence number + issue date + expiry + issuer, and the signatory's name (EN/AR), designation, national ID, passport and visa details | `trade_license` - a company request without one is rejected at once |

A company's signatory (the holder of the uploaded ID, passport or visa) must be one of the people listed on the trade licence; otherwise the request fails with a clear reason.

**Formatting.** Once a request has passed every check, `form_data` values are standardised for both tenant types: gender as `Male` / `Female` (from `M`, `f`, `male`, ... in any case), every date as `DD-MM-YYYY` (e.g. `14-10-2026`), and every country field (nationality, home/resident country, passport issuing location) as an exact country name from `app/data/country_codes.json` - from a 2- or 3-letter code, another name, or a nationality.

**New country names.** A nationality or home/resident country value neither list knows is returned as extracted and saved to `learned_countries/new_country_codes.json` (2-3 letters) or `learned_countries/new_nationalities.json` (anything else). In the background, Gemini gives its 2-letter country code (a short prompt - the country list is never sent) and `app/data/country_codes.json` turns the code into the exact name; the response is never delayed, and from the next request on the value is converted with no model call. Each entry (`{"name": "Syria", "code": "SY", "source": "model", "added": "..."}`, or `"name": null` when it is not a country or its code is not in the list) can be corrected by hand without a restart, and good entries moved into `app/data`. The folder is set by `COUNTRY_LEARNING_DIR`, is not in git, and must survive deployments; `COUNTRY_LEARNING_ENABLED=false` switches learning off.

**How a request is judged.** Every document is downloaded, read, type-checked, paired (a National ID's front and back may be two files) and self-verified - all documents in parallel, and every document is checked even after one fails, so the response shows every problem at once. Cross-verification (do the documents belong to the same person/company?) and the crop/rotate/upload step run only once every document has passed. The result is all-or-nothing: any problem means `status: "failed"`, empty `form_data`, and the reason on that document's `documents[].error`.

**Requests rejected straight away** (no processing, no cost), with a plain-English `detail`:
- an empty `source`, `document_id`, `document_type` or `document_name` (422)
- two documents with the same `document_name` (400)
- a `company` request without a trade licence (400)

**Error messages** are short, plain English: the specific reason first, then what to do - e.g. *"The passport expired on 12 May 2023, before today (6 October 2026). Please upload a valid passport."* Technical details go to the logs only.

### Progress bar (background jobs)

```
POST /api/v1/ocr/leasing/jobs          (or /jobs/upload)  ->  202 {"job_id": "..."}        once per upload
GET  /api/v1/ocr/leasing/jobs/{job_id}                      ->  every 1-2 s until Completed
```

`GET /jobs/{job_id}` always answers:

```json
{ "jobId": "...", "percentage": 65,  "status": "Processing", "result": null }
{ "jobId": "...", "percentage": 100, "status": "Completed",  "result": { ...the normal leasing response... } }
```

- `percentage` combines every document's progress through the stages (download 10, extraction 30, pairing 5, self-verification 20, cross-verification 10, crop/upload 25). It stays at 99 or below while running and becomes 100 when the job ends - successfully or not.
- `result` is always the normal leasing response shape. Whether the documents passed is `result.status` (`success` / `failed`), with reasons in `result.documents[].error`. If the job could not finish for a reason on our side, `result` is a `failed` response with a general friendly message on each document.
- Jobs are held in memory for one hour after finishing, and are lost on a service restart - store the result as soon as it is `Completed`. Polling a job does not count against the rate limit.

## Type-detection endpoints

`POST /api/v1/ocr/leasing/detect` (JSON) and `/detect/upload` (form-data) take only the files (or links), `tenant_type` and `document_refnumber`. Sending `document_type`, `document_name` or `document_id` is rejected, so a caller never believes a declared type was used. `/detect/jobs` and `/detect/jobs/upload` run the same in the background and are polled with the normal `GET /api/v1/ocr/leasing/jobs/{job_id}`.

```json
{
  "tenant_type": "individual",
  "document_refnumber": "TEST-001",
  "sources": [ { "source": "ApplicationFiles/123/EmiratesID.pdf" }, { "source": "ApplicationFiles/123/Passport.jpg" } ]
}
```

**How it works.**
- **Split.** Each file is looked at once as a whole (one extra model call per file) and split into the documents it holds. A file holding one document is used as it is; a file holding several is cut into one file per document. The front and back of one card stay together.
- **Type.** A type is accepted only when two independent readings agree - the split call's and the extraction call's own. The file name is never used. Documents of an unsupported type come back as `ignored` and are not checked. Error messages name a document by its type ("Passport", or "National ID 2" when there are several), so a message about two documents says which two.
- **Tenants.** One request may hold several tenants. Documents are grouped by person (a shared ID number, or matching name and date of birth); for a company, one tenant per trade licence and each listed person whose documents were uploaded. A licence none of whose listed people were uploaded fails - a company never passes with company details alone. Several people's National ID fronts and backs uploaded as separate files are matched by ID number, then by a same-card check.
- **Checks.** Each individual tenant needs their own National ID; a request with none at all is rejected with the usual `400`. Self-verification (expiry included), front/back pairing and cross-verification are the normal pipeline's, with short plain-English messages; each tenant is judged on its own.
- **Files.** Cropping and rotation happen last, only for documents that passed, and each document gets its own file - always a PDF (a front/back pair is one 2-page PDF), named after its tenant and type - also when there is only one tenant: `Tenant_1_National_ID.pdf`, `Tenant_1_Passport.pdf`, `Tenant_2_National_ID.pdf`, and `Tenant_1_National_ID_2.pdf` for a second one of a type (a trade licence shared by two signatories: `Tenant_1_2_Trade_Licence.pdf`). A blob source's file is uploaded beside its original (an original is never overwritten - a clashing name gets `_ocr` added) and `ocr_file` is its blob URL. Every file is also kept on the server in `LEASING_OUTPUT_DIR/<document_refnumber>/`; for an attached file with no blob link, `ocr_file` is that local path.

**Response.** `form_data` and `missing` are always arrays, one entry per tenant, and every document lists the tenants it belongs to:

```json
{
  "status": "success",
  "tenant_type": "individual",
  "document_refnumber": "TEST-001",
  "form_data": [ { "tenant_id": "tenant_1", "status": "success", "tenant_name_en": "...", "...": "..." } ],
  "missing": [ { "tenant_id": "tenant_1", "fields": ["po_box"] } ],
  "processing_time": "21.4 sec",
  "documents": [
    { "tenant_ids": ["tenant_1"], "document_type": "national_id", "status": "valid",
      "merged": false, "sources": [ { "source": "...", "file_name": "EmiratesID.pdf", "pages": [1, 2], "ocr_file": "..." } ],
      "error": null }
  ]
}
```

`status` is `success` only when every tenant succeeded. A document's `status` is `valid`, `invalid` (reason in `error`), `ignored` (unsupported type) or `not_checked` (another file could not be read).

### Settings added for these features (all optional, defaults shown)

| Setting | Default | Purpose |
| --- | --- | --- |
| `RESULT_CACHE_SECONDS` | `10800` | Reuse the extraction (and a passed self-check, same day) of a byte-identical file for this long. Memory only. `0` = off |
| `GEMINI_VERIFICATION_THINKING_BUDGET` | `256` | Thinking tokens for self- and cross-verification - lower is faster |
| `GEMINI_VISION_MODEL` | empty | Separate model for box detection and orientation only; empty = `GEMINI_MODEL` |
| `OUTPUT_RETENTION_HOURS` | `24` | Stored originals and crops older than this are deleted hourly. `0` = keep |
| `LOG_EXTRACTED_RESULTS` | `false` | Writes extracted ID data to `logs/results.log` - development only |
| `ALLOWED_SOURCE_HOSTS` | empty | Extra hosts a `source` URL may point at, besides the blob container |

## Endpoint

`POST /api/v1/ocr` accepts one or more `multipart/form-data` documents:

| Field | Type | Description |
| --- | --- | --- |
| `file` | file | JPG, JPEG, PNG, WEBP, or PDF; for `national_id`/`passport`/`visa`, front and back can be in this one file, or uploaded as two separate files with the same `documentType` in the same batch (see **Endpoint** below). Repeat the field once per document |
| `documentName` | text | Display name used in errors. Repeat once per document |
| `documentType` | text | Any type, e.g. `national_id`, `passport`, `visa`, `driving_license`, `trade_license`. Repeat once per document |

`documentType` accepts **any** document type, not a fixed KYC list. The value is normalised before use — trimmed, lowercased, and spaces/hyphens converted to underscores, so `Trade License` and `trade-license` both become `trade_license` — and must be 1–50 characters of letters, digits, spaces, hyphens, or underscores, since it reaches log lines and the Gemini prompt. These types are checked against what the file actually shows (and self-verified): `national_id`, `passport`, `visa`, `bank_statement`, `ejari_certificate`, `trade_license`, `tenant_form`, `initial_approval`, `salary_certificate` (also sent as Salary Statement, Salary Letter or Payslip), `tenancy_contract` and `cheque` (also Cheque Copy, Security Cheque, PDC). Common spellings are mapped to these (`DOCUMENT_TYPE_ALIASES` in [app/schemas/ocr.py](app/schemas/ocr.py)) - e.g. Emirates ID becomes `national_id`. Any other type is passed through to Gemini as-is, which will populate whichever of the fixed `data` fields it can actually read and leave the rest null.

**Several documents can be sent in one request.** Repeat all three fields once per document; they are paired **by position**, so the first `documentType` belongs to the first `file`, and so on. A request whose three field counts do not match is rejected with a 400 before any document is read.

**The response is always exactly one object, in exactly the schema a single document has always used — never an array, never a new key, no matter how many documents were uploaded.** Each document is validated in upload order — file checks, type match, front/back completeness, self-verification — and the moment any one of them fails, the whole request stops right there: the response is `data` emptied and the reason in `errorInfo`, exactly as a single bad document has always reported it, and any document after the failing one in upload order is never even read. Only once every uploaded document has individually passed does cross-verification run across all of them (see **Validation**); a conflict there stops the request the same way. Only once the whole batch has cleared every one of those gates, including cross-verification, does the crop/rotate pipeline run at all, for every document at once — an error anywhere in the batch, on any document, means none of them is ever cropped or rotated, even one earlier in upload order that had already individually passed. Only then are every document's own fields folded into that one response — a National ID's `National_Id`, a passport's `Passport_Number`, a visa's `Visa_Number`, all landing in the same `data` object beside each other, first non-null value in upload order for any field two documents both happen to supply. A file-validation problem or a provider error (a corrupt file, an unsupported type, Gemini unavailable) keeps its own HTTP status code (400/413/502) exactly as a single-document request always has, whichever document in the list it came from; a type mismatch, an incomplete document, a self-verification finding, or a cross-verification conflict all answer `200` with the problem in `errorInfo`, also unchanged from the single-document contract. Request-level rejections — HTTPS, rate limit, malformed multipart, mismatched field counts — still answer with one error envelope at their own status code, whatever the document count.

**A National ID's front and back no longer have to be in the same file.** If two files in the same batch share a `documentType` from `FRONT_BACK_TYPES` and each is individually missing a side — one shows only the front, the other only the back — the pipeline holds them instead of rejecting either outright: `GeminiService.verify_document_pair` looks at both images and confirms they are genuinely the same document (not two different people's uploads that happen to share a type and a missing side), then the two original files are combined into one PDF and re-extracted as a single document, going through every remaining gate — completeness, self-verification, cross-verification, cropping — exactly as a normal single-file upload would. This only ever resolves the single, unambiguous case: exactly one file showing only the front and exactly one showing only the back. Three or more incomplete uploads of the same type, or two that show the same side, cannot be matched automatically and are rejected with a clear `errorInfo` explaining why, rather than guessing. A pairing confirmed by Gemini but rejected — the two images turn out not to be the same document — is also reported as an error, the same way every other validation finding is. None of this adds a key to the response: the combined result reads exactly like a single well-formed upload's `data` always has. The combined, cropped/rotated document is written to `DETECTION_OUTPUT_DIR` with a `_merged.pdf` suffix — the API response itself never carries a file.

Documents can be uploaded in any order. There is no requirement to send a National ID first.

Rate limiting counts requests, not documents, so uploading ten documents together costs one of `RATE_LIMIT_REQUESTS`. Token usage is likewise accumulated per request: `usage.log` reports what extracting and validating every uploaded document together spent, with `status` `success` or `validation_failed`.

Optional headers: `X-Tenant-ID`, `X-User-ID`, `X-Request-ID`, `X-Detection` (`on`/`off`, overrides `DETECTION_ENABLED`), and `X-Validation` (`on`/`off`, overrides `VALIDATION_ENABLED` — see **Validation**). Requests are rate-limited per IP, tenant, and user identity using `RATE_LIMIT_REQUESTS` within `RATE_LIMIT_WINDOW_SECONDS`.

The extension, declared MIME type, and file signature must agree. PDFs are parsed with strict mode and encrypted, corrupt, empty, or unreadable files are rejected. Images are decoded and verified with Pillow.

**The front/back requirement applies only to `national_id`** - a passport or visa is complete with just its details page (`FRONT_BACK_TYPES` in [app/schemas/ocr.py](app/schemas/ocr.py)). Page or file count alone can't prove front/back presence — both sides routinely share a single page, and a two-page file can just as easily be the same side twice — so completeness for a National ID is Gemini's own call: it reports `Front_Side_Visible`/`Back_Side_Visible` after reading the file. A file missing a side is no longer rejected on the spot: the pipeline holds it and checks whether another file in the same batch, of the same type, supplies the complementary side (see **Endpoint** above) — only once that possibility is ruled out, or a candidate pairing is confirmed and the merged file is *still* incomplete, is `data` dropped and the incomplete error reported. Every other document type — tenancy contract, Ejari certificate, trade licence, and so on — is complete as submitted: a single page is never rejected, and a stray `documentComplete=false` from Gemini is ignored rather than turned into an error, so the rule is enforced in code as well as in the prompt.

Oversized uploads are downscaled before the extraction call. The API rejects any image whose longest edge exceeds 8000px with a 400 (which surfaced as a `502 OCR provider unavailable`), and it internally scales anything above ~1568px down regardless, so a larger original buys no extra detail. `downscale_image_for_api` caps the longest edge at `GEMINI_MAX_IMAGE_EDGE` (1568), re-encoding as JPEG and updating the declared media type to match; PDFs pass through untouched, since their pages are rasterised server-side and are not subject to the pixel limit. A resize failure is logged and the original bytes are sent anyway rather than failing the request. Set it to `0` to disable. The box-detection and orientation calls were never affected — they already downscale to 1024px — and cropping still works from the full-resolution original, so output quality is unchanged.

Gemini receives one robust system instruction and one extraction prompt. Both explicitly treat document text as untrusted data and prohibit embedded instructions from changing extraction rules, schema, security controls, or application behavior. Every field is read from wherever it is clearly printed with its own label, on any page of the file — `National_Id`/`Passport_Number`/`Visa_Number` and the identity fields (`Nationality`, `Date_Of_Birth`, `Tenant_Name_En`/`Ar`, `Gender`) usually come from a National ID, passport, or visa, but an Ejari certificate, tenancy contract, or tenant form routinely prints these same tenant details too, and reading them from there is deliberate, not an oversight (`Visa_Number` specifically from the field labeled "File"/"File Number" on a visa, since residence visas typically have no field literally labeled "Visa Number"). `Occupation`/`Employer_Name` follow a priority rather than a hard ban: prefer a National ID or visa when either is present in the same uploaded file, falling back to the passport only when it's the only document present and shows that information. Nothing is ever copied, inferred, or guessed — only a value genuinely printed with its own label reaches `data`. Every long number gets one extra self-check, deliberately format-agnostic rather than pinned to one document's own layout (a hardcoded shape for an Emirates ID would do nothing for a passport number, a visa file number, or an IBAN): read it twice independently, straight off the image both times, and use only what both readings agree on — a dropped, doubled, or swapped digit is the most common transcription slip in a long number, and re-reading catches it without needing a fixed pattern to check against. Both calls also run at `temperature=0` with a fixed `seed`, favouring the model's single most confident reading of the page over a less certain sampled alternative — the difference between reliably transcribing a number that happens to sit on an unexpected document and missing it. **Gemini is never asked to produce JSON.** It returns plain text only — one `Field_Name: value` line per field, using the exact `DATA_FIELDS` names, with the literal word `null` for anything missing or unreadable, plus a final `Document_Complete: true|false` line. The template in the system instruction is generated from `DATA_FIELDS` itself, so the format the model is asked for cannot drift from the fields the parser knows about.

`parse_extraction_text` in [app/services/post_processing.py](app/services/post_processing.py) turns those lines into the `data` map, and Python builds everything else: the response envelope, `missingInfo`, `errorInfo`, `success`, and `ocrReferenceId`. The parser is deliberately line-tolerant — it matches known field names and skips every other line, so a stray markdown fence, heading, bullet, or sentence of preamble costs one skipped line instead of failing the whole extraction the way a single bad character breaks `json.loads`. Placeholders such as `N/A`, `not visible`, `unknown`, and `-` are normalized to null; a colon inside a value is preserved; and a response containing no recognisable field lines at all is treated as a model failure (502) rather than a silently empty extraction. Model output is never returned directly. Box detection is separate and still uses native structured output, since its coordinates are internal data rather than the caller-facing result. Extended thinking is explicitly disabled (`thinking={"type": "disabled"}`) — this task needs direct extraction, not reasoning, and the SDK's adaptive-thinking default can otherwise consume the whole `GEMINI_MAX_TOKENS` budget before ever emitting the answer.

## Validation

Both checks below are Gemini's own judgment, not a Python rule engine, and neither ever adds a field to the response: a problem found is reported exactly the way every other `/ocr` rejection already is — one more object in `errorInfo`, with `DocumentError`/`DocumentErrorToShow` set to the problem in plain words. Both are controlled by `VALIDATION_ENABLED` (default `true`) and the `X-Validation` header, exactly like `DETECTION_ENABLED`/`X-Detection`; only a verification call's own *operational* failure is swallowed the same way a detection failure is — it never turns an otherwise-good extraction into an error. A genuine finding is never swallowed.

**Every stage can stop the whole request outright, and every one of them empties `data` along with it.** A file-validation problem, a provider error, a type mismatch, an incomplete front/back document, a self-verification finding, or a cross-verification conflict — any one of them, on any one of the uploaded documents — ends the request immediately: nothing past that point runs for any document still waiting in upload order, and the response never carries both a reported problem and the values read alongside it, from that document or any other. That includes the crop/rotate pipeline itself — it never runs until the whole batch has cleared every validation gate, so a problem anywhere in the batch means no document in it is ever cropped or rotated, not even one earlier in upload order that had already individually passed. `success` follows from that the same way it always has: it is `false` whenever `data` is empty, which is now every case with a problem in `errorInfo`.

**Self-verification** runs for every document that reaches extraction as the type it was selected as, before the crop/rotate pipeline is even considered — the crop/rotate pipeline never runs per document at all; it only runs once, across the whole batch, after every document has individually passed *and*, for a batch, cross-verification has too (see below). `GeminiService.verify_document` shows the document image to Gemini a second time and asks it to check what the extraction call itself doesn't: completeness and legibility, required fields actually present and readable, ID/reference number formats, date logic (an issue date before its own expiry, a period's start before its end), whether the document is expired as of today, and internal consistency (a machine-readable zone against the printed text, a front side against a back side, or any other value the document contradicts itself on). Because this call looks at the actual image rather than this service's own narrow `data` fields, it can flag a problem on a bank statement, trade license, or cheque just as well as on a National ID, even though this API never extracts most of those documents' own fields into `data`. If it finds anything, the whole request stops there: `data` is emptied, the crop/rotate pipeline never runs for this document or any other in the batch, and no document later in upload order is even read. Every message it writes is one plain sentence for the person who uploaded the document, not a verification report — no jargon ("MRZ", field names, raw codes) — and every one of them must both say what's wrong and show the actual values that make it wrong: *"The name on the front and back don't match: Ahmed Khan vs Ahmad Khan"*, never a bare verdict like "the name is inconsistent" with nothing to check it against. A date-related finding names both dates, including today's own date — e.g. *"The passport shows an expiry of 12 May 2023, which is before today, 23 September 2026"* — so any finding is checkable at a glance instead of taken on faith. The same rule applies to cross-verification's own findings. It is explicitly told not to flag the document being upside down, sideways, or tilted — that's the crop/rotate pipeline's job (see below), not a real defect — and not to mistake the small greyscale "ghost" photo and day/month stamp on a UAE Emirates ID (part of the holder's own date of birth, printed as a security feature) for a second date that ought to match the card's issuing or expiry date. Unlike extraction, this call and cross-verification both run with a small thinking budget rather than none — comparing values is a judgment call, not a direct transcription, and benefits from the model actually working through it instead of answering in one shot.

An **incomplete front/back document** (see **Endpoint** above) is checked after every document in the batch has been extracted and type-matched, before self-verification or detection ever run. A National ID missing its required other side is not an immediate problem if another file in the same batch, of the same type, supplies it — that pairing is confirmed by Gemini and the two are merged into one document first (see **Endpoint**). Only a document nothing else in the batch can complete, or an ambiguous group of incomplete uploads, or a pairing Gemini does not confirm, stops the request; self-verification and detection are never spent on any of those.

**Cross-verification** runs only when two or more documents were uploaded and every one of them individually cleared type-matching, completeness, *and* self-verification — by the time this stage runs, there is nothing left in the group for it to exclude; a single document skips this stage entirely, since there is nothing to compare it against. `GeminiService.verify_documents_cross` sends Gemini the fields already extracted from every one of them - labelled by `documentName` - on the explicit assumption that every uploaded document belongs to the same person or tenant, and asks it to find every case where two disagree about what should be the same fact: the same person's name, date of birth, nationality, gender, or identifying numbers; the same contact, employment, company, property, tenancy, or financial detail; any of it, whenever it is actually present on two or more of the uploaded documents. A National ID for one person and a Visa for someone else entirely is exactly the case this catches — the prompt is explicit that a name/DOB/ID mismatch is never dismissed as "two unrelated documents". For names specifically, it first checks whether one is just a shorter or longer version of the other — a passport omitting a middle or maternal name a National ID includes is the same person at a different level of detail, not a conflict — and only reports a name finding when a component present on both is genuinely different, or the names share nothing meaningful at all. Any conflict found stops the request the same way as every other stage - `data` stays empty and the conflict is reported in `errorInfo`, and the crop/rotate pipeline never runs for any of the documents in the batch, even ones that had already individually passed. This is the last validation stage in the pipeline; only once it, too, has cleared does the crop/rotate pipeline run at all.

## Postman

Create a `POST` request to `https://api.example.com/api/v1/ocr` in production. Use `http://127.0.0.1:8000/api/v1/ocr` only when `REQUIRE_HTTPS=false` for local testing. When `API_KEYS` is set, add the header `X-API-Key: <key>`.

In **Body > form-data**, add:

| Key | Type | Example |
| --- | --- | --- |
| `file` | File | `national-id-front-back.pdf` |
| `documentName` | Text | `National ID` |
| `documentType` | Text | `national_id` |

Documents can be sent in any order. For a National ID, passport, or visa, front and back can be combined into one PDF or multi-page image file, or sent as two separate `file` entries with the same `documentType` in this same request — see **Endpoint** above. They must still be part of the same request; two separate `/ocr` calls have no way to be matched to each other. Other document types have no front/back requirement at all.

To send several documents at once — merged into one response, and cross-verified against each other — add the three keys again for each one; Postman keeps repeated form-data keys in the order they are listed, which is the order they are paired in:

| Key | Type | Example |
| --- | --- | --- |
| `file` | File | `national-id-front-back.pdf` |
| `documentName` | Text | `National ID` |
| `documentType` | Text | `national_id` |
| `file` | File | `passport.jpg` |
| `documentName` | Text | `Passport` |
| `documentType` | Text | `passport` |

Equivalent curl request (single document):

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/ocr `
  -H "X-API-Key: <key>" `
  -F "file=@C:\docs\national-id-front-back.pdf" `
  -F "documentName=National ID" `
  -F "documentType=national_id"
```

And for two documents in one request, which merges both into the one JSON object below (`National_Id` from the first file, `Passport_Number` from the second, both beside each other):

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/ocr `
  -H "X-API-Key: <key>" `
  -F "file=@C:\docs\national-id-front-back.pdf" `
  -F "documentName=National ID" `
  -F "documentType=national_id" `
  -F "file=@C:\docs\passport.jpg" `
  -F "documentName=Passport" `
  -F "documentType=passport"
```

## Response contract

Every successful Gemini result is normalized to exactly the requested top-level keys. `missingInfo` contains only the specified snake_case names whose corresponding `data` values are null. An incomplete front/back document — only possible for `national_id` — adds an `errorInfo` object with `DocumentName`, `DocumentFileName`, `DocumentError`, and `DocumentErrorToShow`.

`success` reflects whether *any* field was actually extracted — it is `true` as soon as one `data` value is non-null (partial extraction still counts, and for several uploaded documents, so does one contributing a field another didn't), and `false` when every `data` value is null, whether that's because Gemini read the document(s) but found nothing usable, or because a problem anywhere in the pipeline stopped before any data was kept (rejected file, type mismatch, incomplete document, self- or cross-verification finding, rate limit, internal error). It is derived automatically from `data` on every `OcrResponse` (see `_derive_success` in [app/schemas/ocr.py](app/schemas/ocr.py)), not set independently by each response builder — it does not indicate whether the HTTP call itself succeeded, which is what the status code is for.

**Every response from `/ocr` uses this same envelope, including failures, and no matter how many documents were uploaded** — a rejected file, a missing field, a rate-limit hit, an HTTPS requirement, a type mismatch, an incomplete document, a validation finding, or an internal error all return the identical top-level shape (`success`, `data`, `missingInfo`, `errorInfo`, etc.) with the failure reason placed in `errorInfo`, at the original HTTP status code (400/413/422/429/500/502) where one applies, `200` for the validation-stage findings. Callers never need a second code path for a bare `{"detail": "..."}` body, and never need to branch on whether one document or several were sent — the response is always this one object. `DocumentName`/`DocumentFileName` are `"unknown"` only for failures that happen before the multipart body is parsed (HTTPS/rate-limit checks, malformed requests) — anything that fails after a document's own name and file are known reports the real values.

`GET /health` returns `{ "status": "ok" }` and does not call Gemini.

## Document box detection and crop

Every `/ocr` request also runs a second, independent Gemini call — `GeminiService.detect_box_corners` in [app/services/gemini.py](app/services/gemini.py) — concurrently with the main KYC extraction call (`asyncio.gather` in [main.py](main.py); both are real network calls awaited natively, so box detection genuinely runs at the same time as extraction rather than before or after it, and adds no serial latency to the response). A detection/crop failure is logged and never affects the returned OCR result, and this pipeline never changes what is sent to the extraction call.

For each page (a PDF page is rasterized with `pypdfium2` at `DETECTION_PDF_DPI`; an image is used as-is), [app/services/detection.py](app/services/detection.py) downscales a copy for efficiency and sends it to Gemini asking for the four corners of **every distinct physical document in that frame**, as normalized (0–1) fractions of the page's width/height — this captures skew/rotation in the same step, since a quadrilateral's corners encode both position and tilt. Those fractions are then applied to the *original, full-resolution* page to compute pixel coordinates, and a perspective transform straightens and crops each document in one step. If Gemini reports no confident boundary, or an answer is malformed/degenerate (near-zero area), the full frame is used unchanged rather than risking a bad crop — this includes any failure of the Gemini call itself (timeout, API error, invalid JSON).

Each crop is widened by a safety margin on every side, because Gemini's corner estimate is routinely a few pixels inside the true edge and a crop that lands short loses information permanently. The margin is `max(DETECTION_CROP_PADDING_PX, longest edge of the detected document × DETECTION_CROP_PADDING_RATIO)` — 5% of the longest edge by default, with the 10px value acting only as a floor for very small crops. It is scaled off the document rather than fixed in pixels so it means the same thing on a 400px thumbnail and a 2200px PDF render, and derived from the *longest* edge so the short axis gets the same absolute margin as the long one — ID cards are wide, so a proportional miss clips their top and bottom edges first.

All four sides now use the same 12% allowance. An earlier build gave the top a larger share, which was wrong once rotation entered the picture: the bonus was applied in the original frame's coordinates *before* the crop was rotated, so it only landed on the document's real top when the document happened to already be upright. On a rotated document the bonus went to a side and the header — the edge most prone to clipping — kept only the small margin, which is why sideways documents clipped. Since orientation is only known *after* cropping, there is no way to tell in advance which edge will become the top, so the allowance is equal all round. `DETECTION_CROP_TOP_PADDING_RATIO` still exists and is still clamped so the top can never be tighter than the other sides.

The detected quad is mapped to an *inset* rectangle rather than being grown, so the margin is filled with the real pixels just outside the detected edge rather than blank filler. Where a document sits flush against the frame edge there are no pixels to recover and the margin fills with the border colour instead. Set all three padding values to `0` to crop exactly to the detected bounds. The detection prompt separately instructs Gemini to place uncertain corners slightly outside the visible edge rather than inside it.

After cropping, each document goes through a fixed post-processing stage in `enhance_crop`: **rotate, then resize, then enhance**, in that order so the size target applies to the final orientation and the sharpening isn't smeared by a later resize.

*Rotation.* The perspective warp fixes skew but cannot know which way is up — only reading the document tells you that. A dedicated second Gemini call (`detect_orientations`) judges this **on the cropped documents**, not on the original scene, since one isolated document is a far cleaner signal than a document picked out of a full frame. All crops go in one batched call, so document count never multiplies request count. It returns `top_edge` — which edge of the image the document's top currently lies along, as one of `top`/`right`/`bottom`/`left` — plus a small residual `tilt_degrees`, `confidence`, and a one-phrase `reasoning`. **The model is never asked for an angle.** `TOP_EDGE_ROTATIONS` in [app/services/detection.py](app/services/detection.py) maps the edge to the clockwise turn that corrects it (`left` → 90, `right` → 270) and adds the tilt; code then applies it with `cv2.rotate` and records it in the metadata JSON. The split exists because the arithmetic, not the perception, was what failed: repeat runs on a byte-identical crop agreed on what they saw and still returned opposite quarter turns in half of all runs, since a structured answer commits to its first field before any reasoning is written. A `tilt_degrees` outside ±45° is dropped rather than added, since that is a quarter turn `top_edge` already carries, and anything under ±5° is treated as zero — a crop that has already been perspective-warped onto its own corners has no real slant left, but the model reports a degree or two on almost every document, which would push every quarter turn off-grid onto `cv2.warpAffine`'s resampling path instead of `cv2.rotate`'s lossless one. **There is no sampling control available:** this model rejects `temperature` outright (``temperature` is deprecated for this model`) and the SDK exposes no `top_p`, `top_k`, or seed, so repeatability comes from constraining the answer — four enum values and a deadbanded tilt — rather than from decoding settings. Anything off-grid or unparseable means "leave it alone" rather than a guessed turn, and a failure of the call degrades to saving the crops unrotated rather than losing them. Crops whose confidence is under `DETECTION_ROTATION_MIN_CONFIDENCE` (0.7) are applied anyway but logged as warnings. Corner order from box detection is irrelevant, since `_order_box_points` re-sorts corners in image space.

The angle is **not restricted to quarter turns**. Gemini reports the coarse orientation *plus* any remaining tilt as a single combined clockwise figure — an upside-down card that also slopes 10° answers 190 — and OpenCV applies it: exact quarter turns go through `cv2.rotate` (a lossless transpose/flip), anything else through `cv2.getRotationMatrix2D` + `warpAffine` with the canvas grown to the rotated bounding box so no corner is cut off. This matters because the perspective warp only removes tilt when the detected corners are accurate, and the model's corner estimate is not always accurate; letting the rotation carry a fine angle recovers documents the warp left slanted.

**Corners are refined before cropping.** Gemini's corner coordinates are semantically right — it reliably says *which* region holds a document — but geometrically approximate: on a tilted Aadhaar sample the reported quad sat ~35° off the card's true edges. A wrong quad makes the perspective warp bake in a shear instead of removing one, which surfaces as a crop that is still tilted, ringed with background, and wedged with border fill. `refine_box_with_edges` therefore searches for the strongest contour *inside* each region Gemini reports and snaps the quad onto the real border, keeping the model's semantics and replacing only the geometry. The fit is rejected, leaving the original box untouched, if no convincing contour is found or if its area bears no sensible relation to the reported region. Refined detections are marked `gemini_vision+edges` in the metadata, so you can tell which geometry was used.

A refined fit is only accepted when it clears two guards, because a similar *area* is no evidence that a box is right — a clipped or shifted quad can match the original's size closely. First, **edge support**: each candidate is scored by how strongly its weakest side lies on real image gradient, and the refinement is discarded unless it traces the border better than the model's box. A quad that cuts through the middle of a card has an interior side with almost no gradient beneath it, which this exposes immediately. Second, **no sibling overlap**: two documents in one frame never overlap, so a fit that reaches into a neighbour's region has locked onto the wrong thing. That case is common when cards sit side by side with only a thin gap, since the search window necessarily spans it. Both guards fall back to the model's box, which is often already correct.

Input images are loaded through `ImageOps.exif_transpose`, so the EXIF orientation tag is honoured. Phone cameras and scanners routinely store pixels one way and a tag saying how to display them; viewers apply it and PIL does not. Without this the pipeline would analyse a differently-oriented image than the one you are looking at, and reported rotations and left/right labels would not line up with the file on screen.

This is what makes the rest behave: with accurate corners the warp fully deskews, so the orientation pass usually only needs a clean quarter turn through the lossless `cv2.rotate` rather than an arbitrary-angle warp, and the crop lands tight on the document. Because the corners now sit on the real border, `DETECTION_CROP_PADDING_RATIO` drops to 1.5% — the old 12% margin existed only to absorb inaccurate corners and showed up as unwanted background. Refinement operates per region, so two cards in one frame stay two separate, individually-snapped crops.

**Every rotation is logged** to `logs/rotation.log`, one JSON object per cropped document: the angle requested with its confidence and reasoning, whether it was applied, which OpenCV path ran (`cv2.rotate` vs `cv2.warpAffine`), the detected box angle and method, and the crop size before and after. That makes a bad rotation auditable after the fact without re-running anything.

*Resolution and clarity.* `DETECTION_CROP_TARGET_LONG_EDGE` (1600 by default) normalizes every crop to a moderate, consistent size — small crops are enlarged with cubic interpolation, oversized ones reduced with `INTER_AREA`. Upscaling is capped at 3× because enlarging invents no real detail; a tiny crop is made legible rather than blown up into a soft imitation of high resolution. `DETECTION_CROP_CONTRAST_CLIP` (1.5) applies CLAHE to lightness only, lifting contrast in shadowed or unevenly lit scans without shifting the document's colours, and `DETECTION_CROP_SHARPEN_AMOUNT` (0.6) applies an unsharp mask at final resolution. Measured on a real crop at matched resolution, the defaults raise edge energy by roughly 60% while keeping clipped pixels near 1.7%. Set any of the three to `0` to skip that step. Note that if crops from PDFs look soft, raising `DETECTION_PDF_DPI` adds genuine detail where upscaling cannot.

One frame often holds more than one document — a card's front and back photographed side by side, or two cards laid out in a single scan. Each is detected, cropped, and saved **separately**, with its own coordinates and a short label (`national id front`, `passport bio page`, …); the prompt explicitly forbids merging two documents into one box. So a single uploaded image can produce two or more cropped files.

Output is written to `DETECTION_OUTPUT_DIR` (`detections/` by default), named after the source file. A single detected document is saved as `<original stem>_<sha256[:8]>.png`; when an upload yields more than one crop (multiple pages, multiple documents per page, or both), each is saved as `<original stem>_<sha256[:8]>_page{n}_doc{m}.png`. One `<original stem>_<sha256[:8]>.json` per upload accompanies them, grouped by page — each page carries `documents_found` plus a `documents` array, and each entry has `region_number`, `label`, `box` (4 corners), `axis_aligned_bbox`, an approximate `angle_degrees`, the `rotation_degrees` quarter turn that was applied to bring it upright, `method` (`gemini_vision`, `fallback_full_frame`, or `fallback_invalid_response`), and its `cropped_file` name. The filename is sanitized before use, so path separators or `..` in an uploaded name cannot escape the output directory.

Note this means every `/ocr` request makes two Gemini calls instead of one — extraction and box detection — which roughly doubles per-request Gemini cost and API-side rate-limit consumption, even though wall-clock latency stays close to the slower of the two since they run concurrently.

**This persists a derivative of the uploaded document image to local disk.** A background sweep deletes it after `OUTPUT_RETENTION_HOURS` (see the retention policy below); restrict access to the folder in the meantime, or disable this pipeline if cropped documents must never be written to disk.

## Retention policy

The defaults are intentionally data-minimizing and are configured in `.env`:

| Item | Retention | Behavior |
| --- | ---: | --- |
| Uploaded document bytes (`/api/v1/ocr`) | 0 seconds | Kept in memory only and released after validation/Gemini processing |
| Leasing originals, crops, merged PDFs | `OUTPUT_RETENTION_HOURS` (24 by default) | Written to `LEASING_OUTPUT_DIR` (`postman_output/`), deleted by the hourly sweep |
| Cropped document image | `OUTPUT_RETENTION_HOURS` (24 by default) | Written to `DETECTION_OUTPUT_DIR` (`detections/`), deleted by the hourly sweep |
| Extracted data | 0 seconds | Not logged unless `LOG_EXTRACTED_RESULTS=true` (development only), in which case `logs/results.log` keeps it for `LOG_RETENTION_DAYS` |
| API response | 0 seconds | Not cached or stored by this app |
| Application logs | 30 days | Set `LOG_RETENTION_DAYS=30`; the local rotating handlers delete older log backups automatically |

Do not enable request-body logging in IIS, nginx, Uvicorn, Postman, or any monitoring agent. If business requirements require persistence, use encrypted storage with a documented deletion job and access audit.

The retention sweep runs at startup and then hourly inside the service; `OUTPUT_RETENTION_HOURS=0` disables it and keeps files forever. Results uploaded to Azure Blob Storage are not touched — set a lifecycle-management rule on the container for those.

**Never set `LOG_EXTRACTED_RESULTS=true` where real customer documents are processed.** It writes every extraction's National ID, DOB, and passport/visa numbers to `logs/results.log` in plain text.

**Source URLs.** `POST /api/v1/ocr/leasing` only fetches from the configured blob container, plus any host listed in `ALLOWED_SOURCE_HOSTS`. Redirects are not followed and downloads stop at `MAX_FILE_SIZE_MB`, so the endpoint cannot be used to make the server reach internal addresses.

## Deployment (Windows VM with NSSM)

The service runs as a Windows service through [NSSM](https://nssm.cc), which starts it at boot and restarts it after a crash. On the VM, in an **elevated** PowerShell:

1. Install the prerequisites: Python 3.12+ (python.org installer, "Add to PATH"), Git, and NSSM (`winget install NSSM.NSSM` or `choco install nssm`, or download nssm.exe and pass `-NssmPath`).
2. Clone and configure:

   ```powershell
   git clone <repo-url> C:\apps\uae-ocr-api
   cd C:\apps\uae-ocr-api
   copy .env.example .env
   notepad .env
   ```

   Set at least `GEMINI_API_KEY`, `GEMINI_MODEL`, `API_KEYS` (a long random value per client, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`), `AZURE_STORAGE_*` if the leasing endpoints use blob storage, and `DOCS_ENABLED=false` if the host is internet-facing.
3. Install and start the service:

   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1
   ```

   This creates `.venv`, installs `requirements.txt`, and registers the `UaeOcrApi` service (auto start, restart on exit after 5 s, console output in `logs\service-stdout.log` / `logs\service-stderr.log`, rotated at 10 MB). Rerunning it reinstalls the service cleanly.
4. Check it: `Invoke-RestMethod http://127.0.0.1:8000/health` should return `{"status":"ok"}`.

**Exposing it.** Keep `APP_HOST=127.0.0.1` and put IIS (URL Rewrite + ARR), nginx or Caddy in front for TLS, with `REQUIRE_HTTPS=true` and the proxy forwarding `X-Forwarded-Proto`. `FORWARDED_ALLOW_IPS` must list the proxy's IP (`127.0.0.1` when it runs on the same VM), or the forwarded scheme is ignored and every request is rejected as plain HTTP. Raise the proxy's request-body limit to at least `MAX_FILE_SIZE_MB` times the number of files per request, and its timeout above `GEMINI_TIMEOUT_SECONDS`. Only for a closed internal network with no proxy: set `APP_HOST=0.0.0.0` and `REQUIRE_HTTPS=false`, and open the port in Windows Firewall (`New-NetFirewallRule -DisplayName "UAE OCR API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`).

**Day to day:**

| Task | Command |
| --- | --- |
| Deploy the latest `main` | `powershell -ExecutionPolicy Bypass -File deploy\windows\update-service.ps1` |
| Status / restart / stop | `nssm status UaeOcrApi` · `nssm restart UaeOcrApi` · `nssm stop UaeOcrApi` |
| Changed `.env` | `nssm restart UaeOcrApi` (settings are read at startup) |
| Startup errors | `Get-Content logs\service-stderr.log -Tail 50` |
| Remove the service | `powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-service.ps1` |

The service runs one worker on purpose: the rate limiter and per-request usage tracking live in process memory.

## Production operations

- Place IIS/nginx/Caddy or an API gateway in front of Uvicorn for TLS certificates, request-size limits, firewall rules, and security headers.
- Allow inbound traffic only from the reverse proxy/security group and restrict outbound traffic to Gemini API endpoints where practical.
- Use separate API keys and `.env` files for development, staging, and production; rotate and revoke keys regularly.
- Dependencies are pinned in `requirements.txt`; run `pip-audit` or an approved dependency scanner in CI and review updates regularly.
- Monitor request latency, success/failure rate, Gemini errors, validation failures, rate-limit events, rejected files, and token/cost usage. On Windows, IIS Advanced Logging plus Windows Event Viewer, Task Scheduler, and Performance Monitor can provide local access/error/latency monitoring; forward sanitized metrics to the approved monitoring system.
- Keep `REQUIRE_HTTPS=true` in production and never publish Uvicorn directly to the internet.

## Development notes

The rate limiter is intentionally in memory for development/testing. Restarting the process clears it. For production, replace it with authenticated, shared state such as a protected database or internal service, and enforce tenant/user identity at the gateway rather than trusting arbitrary headers.