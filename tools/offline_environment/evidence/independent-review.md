# Independent code review — 2026-10-16

Reviewer: separate read-only agent `/root/offline_review`, requested by the
user's independent-review requirement. Not a consensus vote or scientific result.

Independently checked: 15 wheel hashes/sizes and requirements hash, existing
paper venv `pip --isolated check`, three shell scripts `bash -n`, font-only
configuration diffs, platform/overwrite/no-index/hash guards. No target-host,
GPU or full paper rebuild was performed by this reviewer.

Findings and owner fixes:

1. Existing-environment test depended on a preexisting venv and could install on
   fresh clone. Fixed using a temporary ROOT + explicitly precreated directory;
   reviewer read back the fix.
2. Some platform-specific tests could fail misleadingly on Python3.11. Test
   class now explicitly skips incompatible runners.
3. Wheel-lock helper cannot infer official provenance from local files. Default
   now says origin unknown; paper wheels separately matched official PyPI JSON
   digests and source URLs in `paper-pypi-provenance.json`.
4. Debian resolver uses moving indexes. Documented as a re-resolution recipe;
   additionally provided `download_locked_system.py`, which fetches exact pinned
   bytes from the official archive and refuses version/hash substitution.

Reviewer confirmed the old default `evidence_register.tex` file is absent;
changing defaults to four existing sources is a supported build-entry repair.
Acceptance scope: paper Python bundle on its declared platform; not complete
system/native baseline/target-host offline readiness. New locked Debian helper
and baseline replay added after the review are owner-tested, not independently
certified by that original review.
