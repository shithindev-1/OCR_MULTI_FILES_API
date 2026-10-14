"""Type detection: the /ocr/leasing/detect endpoints, which find each document's type themselves.

The caller sends only files (or links), a tenant_type and a document_refnumber - no document type,
name or id. Each file is split into the documents it holds (segmentation.py), and each
document then goes through the same `process_batch` as the normal leasing endpoints, with the type
the segmentation found standing in for a declared one:

  - A type counts only when two independent readings agree: the segmentation's, and the extraction
    call's own Document_Types_Present. Anything else rejects that document.
  - Documents of a type outside the supported list ("other") are ignored: reported, never checked.
  - The mandatory document - a National ID for an individual, a trade licence for a company - is
    checked once the types are known, with the same message as the normal endpoints.
  - Self-verification (expiry included), front/back pairing, cross-verification, crop/rotate and
    upload are exactly the normal pipeline's.

Each document is named for messages by the type found - "Passport", or "National ID 2" when there
are several of a type - never by its file name, which may say anything. Its processed (cropped,
rotated or merged) copy is uploaded beside the original under its tenant and type:
Tenant_1_National_ID.pdf, Tenant_1_Passport.pdf, Tenant_2_National_ID.pdf - always a PDF, and also
kept locally under that name.

Modules:
  - service.py       runs a request: split the files, run the leasing batch, build the response
  - segmentation.py  splits one file into the documents it holds
  - tenants.py       groups the read documents into tenants (people, or companies and signatories)
  - responses.py     the response shape: form_data and missing per tenant, one entry per document
"""
