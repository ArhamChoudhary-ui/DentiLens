# Dataset manifests

`dentalguard-splits.json` preserves the recording groups used in the prior
DentalGuard study. The versioned build records archive and image SHA-256,
taxonomy and annotation versions, provenance, preprocessing, exclusions and
perceptual distance checks. Files containing local archive media or generated
previews belong under the ignored runtime directory. Generate each version
into a new output directory so old manifests remain reproducible.
