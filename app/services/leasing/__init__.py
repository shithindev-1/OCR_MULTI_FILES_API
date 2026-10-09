"""Batch processing behind the /ocr/leasing endpoints.

Both endpoints reduce their input to the same list of items and hand it to `process_batch`, which
reuses the shared OCR pipeline (app.services.ocr_pipeline) step by step rather than
re-implementing any of it, so the leasing and /ocr endpoints cannot drift apart.

How a batch is judged:
  - Every document is checked in full before anything is decided - download, validation,
    extraction and type check, front/back pairing (per document type), self-verification - so one
    document's problem never stops the others from being checked, and the caller sees every problem
    at once. Each failing document carries its own reason in its own `error`; every document that
    passed has `error: null`.
  - Cross-verification runs only once every document passed on its own, and its findings go on the
    documents they name.
  - The *result* is all-or-nothing: any failure blanks `form_data`, sets `status` to "failed", and
    each failing document's reason stays in its own `error` - there is no top-level copy. Cropping, rotation and the
    merged PDF only happen when the whole request passed.
  - The one exception to checking everything is the OCR provider itself being unavailable, which
    fails the request once the extraction stage ends.
  - Within each stage - download, extraction, self-verification, crop/rotate/upload - the documents
    are processed concurrently (at most `app.services.concurrency.MAX_PARALLEL_DOCUMENTS` at once), since none of them
    depends on another. The stages themselves still run strictly in order, so nothing runs that a
    one-by-one run would have skipped: cross-verification still waits for every self-check, and
    cropping still waits for the whole request to pass.
  - `document_name` must be unique within a request: it is how a merged front/back result is traced
    back to its two original documents.
  - Output files (originals, crops, merged PDF) are written under `settings.leasing_output_dir`.
  - A source inside the configured blob container (a path like "ApplicationFiles/.../Passport.jpg",
    or a full URL into that container) is fetched from blob storage, and - only when the whole
    request passed - its cropped/rotated result is uploaded back into the same folder as
    `<name>_ocr.<ext>`, reported in that source's `ocr_file` as its full blob URL. One crop keeps the original's image
    format; several crops become one PDF; a merged front/back pair uploads one PDF named after the
    front. A failed upload is reported on its document and fails the request.

Modules:
  - batch.py           process_batch and process_sources: the stages, in order
  - rules.py           request checks made before anything is downloaded
  - sources.py         downloading a source from an allowed URL or the blob container
  - pairing.py         front/back pairing, per document type
  - reports.py         the response and its per-document entries
  - outputs.py         output folders, and the processed file uploaded beside each original
  - signatory.py       a company signatory's role from the trade licence
  - tenant_results.py  several tenants in one request (type-detection endpoint)
"""
