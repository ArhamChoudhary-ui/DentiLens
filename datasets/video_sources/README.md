# Video source registry

The checked-in JSON files are acquisition leads with unverified rights.
They contain no source media, local paths or identity records. Keep the
operational registry under ignored runtime storage. Use one JSON per local
source and register it through `scripts/video_benchmark.py register`.

An authorized record requires `id`, `name`, `reference`, `category`,
`recording_group`, `split`, `status`, `path`, `license`,
`rights_reference`, `rights_reviewer`, and explicit `permissions`.
Record creator, date and protected patient group where available. Set
`identifiable: false` only after review. Otherwise supply
`consent.reference` and `consent.permissions` for the actual purposes.
Use `expires` for time-limited grants and update the checked local registry
when permission is withdrawn. Set `cfr_verified` after verifying fixed FPS.
Final test sources also require `previously_observed: false`.

Never copy confidential authorization evidence or source video into Git.
