# Legal notes (India) — orientation only, not legal advice

Have a cyber-law practitioner or your college's legal cell review anything you claim in a report or
product. Laws and rules change; check the current text before relying on a section number.

## Where the payload and manifest may help
- **Information Technology Act, 2000** – provisions commonly invoked for misuse of someone's photos
  include identity theft (s. 66C), cheating by personation using a computer resource (s. 66D),
  violation of privacy (s. 66E), and publishing obscene / sexually explicit material electronically
  (ss. 67, 67A).
- **Bharatiya Sakshya Adhiniyam, 2023 (BSA)** – electronic records are admissible subject to the
  conditions of **s. 63** (the successor of s. 65B of the Indian Evidence Act), including a certificate
  about how the record was produced. Practical implication: keep the original file, the protected file,
  the signed manifest and a record of how/when they were created, so a s. 63 certificate can be
  prepared. The I3 payload is *supporting technical evidence*, not proof on its own.
- Complaints: National Cyber Crime Reporting Portal (cybercrime.gov.in) and helpline **1930**.

## Privacy obligations if you build an app
- **Digital Personal Data Protection Act, 2023 (DPDP)** – a face photo and an owner id are personal data.
  The prototype only embeds a keyed tag and stores a SHA-256 of the owner id in the manifest; keep it that
  way, process on-device where possible, and get explicit consent if you ever collect photos (including
  for your evaluation dataset).
- Don't put real people's photos in a public GitHub repo; use public-domain samples or consented data.

## Wording for your report
"I3 embeds a keyed, error-corrected ownership tag and produces an Ed25519-signed provenance manifest,
intended as supporting technical evidence for complaints under the IT Act, 2000; admissibility of
electronic records is governed by s. 63 of the Bharatiya Sakshya Adhiniyam, 2023."
