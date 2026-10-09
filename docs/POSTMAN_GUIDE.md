# Testing through Postman

This is the same OCR pipeline as the existing `ocr_dev_gemini` project - extraction, front/back
pairing and merging, self-verification, cross-verification, cropping, rotation - reachable through
two new endpoints on the same running app:

- **`POST /api/v1/ocr/leasing/upload`** - multipart **form-data**, documents attached directly as files in
  Postman, several at once. **Use this one** - it's what the rest of this guide walks through.
- **`POST /api/v1/ocr/leasing`** - JSON body, documents named by URL instead (the API downloads them
  itself) - covered near the end.
- **`POST /api/v1/ocr/leasing/jobs`** (or **`/jobs/upload`**) then **`GET /api/v1/ocr/leasing/jobs/{job_id}`** -
  the same processing in the background, for a progress bar - see **6. Progress bar** at the end.

Both return the exact same response shape. See `app/services/leasing.py` for exactly what either does and
does not change; the original `POST /api/v1/ocr` (the existing multipart endpoint) still works exactly as
before, untouched.

## 1. Set up

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in your own `GEMINI_API_KEY`/`GEMINI_MODEL`. `.env` is never
committed to git.

## 2. Run the server

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Set `REQUIRE_HTTPS=false` in `.env` for local testing, otherwise the API rejects a plain
`http://localhost` request from Postman. The HTTPS and rate-limit checks apply to every route under
`/api/v1/ocr`, leasing included. If `API_KEYS` is set, add an `X-API-Key: <key>` header to every
request (Postman: **Headers** tab).

## 3. Set up the request in Postman (form-data, multiple files)

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/ocr/leasing/upload`
- **Body tab** → select **form-data**

Add these rows. `file` is type **File** (click the dropdown on the right of the key field to switch
a row from Text to File); everything else is type **Text**:

| Key | Type | Value (example) |
|---|---|---|
| `tenant_type` | Text | `individual` |
| `document_refnumber` | Text | `93ade000-37d4-41f0-bbc7-422dcbe46ad0` |
| `file` | **File** | *(click "Select Files", pick your first document)* |
| `document_id` | Text | `16` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `Nationalid1` |

Click **Send**. That's it for one document.

### Uploading multiple files at once

Add **one more row for each of `file`, `document_id`, `document_type`, `document_name`**, still
using those exact same key names - Postman lets you repeat a key across several rows, and they are
matched up **by position**: the 2nd `file` row pairs with the 2nd `document_id` row, the 2nd
`document_type` row, and the 2nd `document_name` row, and so on. So for a National ID's front and
back plus a passport, the form-data tab ends up with:

| Key | Type | Value |
|---|---|---|
| `tenant_type` | Text | `individual` |
| `document_refnumber` | Text | `93ade000-...` |
| `file` | File | *National ID front.jpg* |
| `document_id` | Text | `16` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `National ID Front` |
| `file` | File | *National ID back.jpg* |
| `document_id` | Text | `17` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `National ID Back` |
| `file` | File | *Passport.jpg* |
| `document_id` | Text | `18` |
| `document_type` | Text | `Passport` |
| `document_name` | Text | `Passport` |

`tenant_type` and `document_refnumber` only ever appear once, at the top - they describe the whole
request, not any one document. Send it: the National ID front and back are detected and merged
automatically (see the existing README's **Endpoint** section for how that works), and all
documents in the request are cross-verified against each other.

If the row order gets mixed up while editing, that's fine as long as each group of 4
(`file`/`document_id`/`document_type`/`document_name`) keeps the same relative order as the others -
Postman sends form-data fields in the order the rows appear, which is what "by position" pairs on.

### Response shape

Success and failure both return this same shape. A successful National ID front + back (merged)
plus a passport:

```json
{
  "status": "success",
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "form_data": {
    "tenant_name_en": "Ahmed ...",
    "national_id": "784-...",
    "...": "...every other field, null if not read..."
  },
  "missing": ["visa_number", "..."],
  "processing_time": "14.72 sec",
  "documents": [
    {
      "document_type": "national_id",
      "document_type_is_correct": true,
      "merged": true,
      "sources": [
        {"source": "nid_front.jpg", "document_id": "16", "document_name": "NID Front",
         "document_type": "National Id", "pages": [1]},
        {"source": "nid_back.jpg", "document_id": "17", "document_name": "NID Back",
         "document_type": "National Id", "pages": [1]}
      ],
      "error": null
    },
    {
      "document_type": "passport",
      "document_type_is_correct": true,
      "merged": false,
      "sources": [
        {"source": "passport.pdf", "document_id": "18", "document_name": "Passport",
         "document_type": "Passport", "pages": [1, 2]}
      ],
      "error": null
    }
  ]
}
```

A failure - here `/api/v1/ocr/leasing` with a `source` that is not a valid link:

```json
{
  "status": "failed",
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "form_data": {"tenant_name_en": null, "...": null},
  "missing": ["tenant_name_en", "...every field..."],
  "processing_time": "0 sec",
  "documents": [
    {
      "document_type": "unknown",
      "document_type_is_correct": false,
      "merged": false,
      "sources": [
        {"source": "not-a-link", "document_id": "16", "document_name": "Nationalid1",
         "document_type": "National Id", "pages": []}
      ],
      "error": "'not-a-link' is not a valid file link or file path. Please check it and try again."
    }
  ]
}
```

An **empty** `source`, `document_id`, `document_type` or `document_name` never gets this far: the
request is rejected straight away with `422` and a plain sentence, e.g.
`{"detail": "The file link for 'Nationalid1' is empty. Please add the document's link or file path and try again."}`

- **Every document is checked in full**, even when another one has already failed: validation,
  type check, front/back pairing, self-verification. Cross-verification (do the documents agree
  with each other?) runs only once every document has passed on its own.
- `status`: whether the *whole request* succeeded. This is all-or-nothing - any document's
  problem empties `form_data` for everyone in the batch. Each failing document's reason is in its
  own `documents[].error`; there is no top-level `error`.
- `documents[]`: one entry per processed document. A front and back sent as two files and merged
  become **one** entry with `merged: true` and both files under `sources`. They stay as two
  separate entries if the request failed before or during merging.
- `documents[].document_type`: the normalised type (`national_id`, `passport`, ...); `"unknown"`
  if that file never got past download/validation. `sources[].document_type` is what you sent.
- `documents[].document_type_is_correct`: `true` once the file was confirmed to be the type you
  selected; `false` on a type mismatch or if it was never checked.
- `documents[].error`: that document's own problem, or `null` if it passed. A front/back pair
  that doesn't match shows the error on both sides; a cross-verification finding shows on the
  documents it mentions.
- `sources[].pages`: page numbers read from that file - `[1]` for an image, `[1, 2, ...]` for a
  PDF, `[]` if the file was never read.
- Output files are not listed in the response - see the next section for where they are.

## 4. Where the files are saved

Nothing is uploaded anywhere - everything lands under `postman_output/` next to wherever you run
`uvicorn` from, organised like this:

```
postman_output/
  <document_refnumber>/
    <document_id>_<document_name>/
      original_<uploaded filename>         <- the exact file you attached in Postman
      <stem>_<sha8>.json                   <- crop/rotation metadata (only if detection is on)
      <stem>_<sha8>.png                    <- final cropped AND rotated page(s) - one file per side
    merged_<front_id>_<back_id>/           <- only for a front+back pair the pipeline merged
      <stem>_<sha8>.json
      <stem>_<sha8>.png                    <- one per side (front, back)
      <stem>_<sha8>_merged.pdf             <- the two sides combined into one downloadable PDF
```

There is no separate "rotated" file: this pipeline crops and rotates a page in one step, so the
`.png` file *is* the final, upright, cropped result. If a National ID's front and back were sent as
two separate files, only the shared `merged_<front_id>_<back_id>/` folder gets crop/rotate/merge
output - the two `<document_id>_<document_name>/` folders each still keep their own
`original_...` file for traceability, since that is what was actually uploaded for each one.

## 5. Common problems

- **`422`** with *"Some required information is missing or not valid: …"*: a form-data row is
  missing or misnamed - check `tenant_type` and `document_refnumber` are present.
- **`422`** with *"The file link for '…' is empty…"* (or document type / document_id / name): fill in
  the empty field.
- **`400`** *"Every file needs its own document_id, document_type and document_name…"*: the row
  counts don't match - e.g. 3 `file` rows but only 2 `document_id` rows.
- **`400`** *"A trade licence is required for a company…"*: `tenant_type` is `company` but no
  `trade_license` document was sent.
- **`status: "failed"` with a plain-language `error`**: this is the pipeline itself finding a real
  problem (wrong document type, a missing side with no matching pair, a self- or cross-verification
  finding) - same as the existing endpoint, just reported in this response shape instead.

## Alternative: JSON body with URLs (`POST /api/v1/ocr/leasing`)

If a document is already hosted somewhere reachable and you'd rather send its URL than attach the
file itself, `POST /api/v1/ocr/leasing` takes the same request shape you originally described, as raw JSON
(Content-Type: application/json, set automatically by Postman for "raw" + "JSON" body):

```json
{
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "sources": [
    {
      "source": "https://<account>.blob.core.windows.net/<container>/ApplicationFiles/LeasingRequestOCR/1/93ade000-37d4-41f0-bbc7-422dcbe46ad0/Nationalid 1.jpg",
      "document_id": "16",
      "document_type": "National Id",
      "document_name": "Nationalid1"
    }
  ]
}
```

Same response shape, same local file layout, same all-or-nothing behaviour - just add more objects
to `sources` for multiple documents. This assumes every `source` URL is directly downloadable with a
plain GET; if yours need a SAS token or auth header, that has to already be part of the URL you send. A URL outside the configured blob container is only fetched when its host is listed in
`ALLOWED_SOURCE_HOSTS` in `.env`.

## 6. Progress bar (background jobs)

The same processing, but in the background, so a progress bar can be shown while it runs.

**Request 1 - start (once per upload)**

- `POST http://127.0.0.1:8000/api/v1/ocr/leasing/jobs` with **Body -> raw -> JSON** (same body as above),
  or `POST http://127.0.0.1:8000/api/v1/ocr/leasing/jobs/upload` with **Body -> form-data** (same rows as section 3)
- Response `202`: `{ "job_id": "cb096b44-47dc-4eb2-8dd7-a8ea7aec8c42" }`
- Optional - save it automatically: **Scripts -> Post-response**:
  `pm.collectionVariables.set("job_id", pm.response.json().job_id);`

**Request 2 - progress and result (send every 1-2 seconds)**

- `GET http://127.0.0.1:8000/api/v1/ocr/leasing/jobs/{{job_id}}`
- While running: `{ "jobId": "...", "percentage": 65, "status": "Processing", "result": null }`
- When done: `{ "jobId": "...", "percentage": 100, "status": "Completed", "result": { ...same response as above... } }`

Stop when `status` is `Completed`, then read `result`: `result.status` is `success` or `failed`, and
any problems are in `result.documents[].error`.

Notes:
- Keep the same `job_id` for all polls of one upload; a new upload gets a new `job_id`.
- `{{job_id}}` only works if both requests are in the same collection and the variable is set -
  hover over it in the URL to check. A 404 *"This job was not found or has expired"* usually means
  the variable was not set, or the server restarted (with `--reload`, saving a file restarts it).
- Requests that can never succeed (an empty field, a company without a trade licence) are rejected
  by Request 1 itself, with no job created.

## 7. Type detection - no document type, name or id (`/api/v1/ocr/leasing/detect`)

The same checks, but the service finds each document's type itself. Send only the files (or links),
`tenant_type` and `document_refnumber`; a request that also sends `document_type`, `document_name`
or `document_id` is rejected.

**Background job, raw JSON body (recommended)**

- Request 1: `POST http://127.0.0.1:8000/api/v1/ocr/leasing/detect/jobs` with **Body -> raw -> JSON**:

  ```json
  {
    "tenant_type": "individual",
    "document_refnumber": "TEST-001",
    "sources": [
      { "source": "ApplicationFiles/123/EmiratesID.pdf" },
      { "source": "ApplicationFiles/123/Passport.jpg" }
    ]
  }
  ```

  Response `202`: `{ "job_id": "..." }`
- Request 2: `GET http://127.0.0.1:8000/api/v1/ocr/leasing/jobs/{{job_id}}` - the same polling
  endpoint as section 6 - until `status` is `Completed`.

Each `source` is a blob path in the configured container (or its full blob URL). One file may hold
several documents, and one request may hold several tenants (for example a husband's and a wife's
National IDs).

**Without a job:** `POST /api/v1/ocr/leasing/detect` with the same body answers directly.
**Form-data:** `POST /api/v1/ocr/leasing/detect/upload` (or `/detect/jobs/upload`) with
`tenant_type`, `document_refnumber` and one `file` row per file.

**Reading the result**

- `form_data` and `missing` are arrays - one entry per tenant, each with its `tenant_id`.
- Each `documents[]` entry has `tenant_ids`, the `document_type` found, and a `status`: `valid`, `invalid` (reason in `error`), or `ignored` (not a supported type).
- `status` at the top is `success` only when every tenant succeeded.
- `ocr_file` is the processed copy, uploaded beside the original and named after its type -
  `National_ID.pdf`, `Passport.jpg`, `National_ID_2.pdf`; with several tenants, `Tenant_1_National_ID.pdf`,
  `Tenant_2_National_ID.pdf`.
- No National ID anywhere in an individual request (or no trade licence in a company request) ends
  the job with that message on every document.
