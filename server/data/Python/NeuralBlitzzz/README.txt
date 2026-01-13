
OSM-TriPH Minimal Python Reference (prototype)
==============================================

Contents
--------
- sss_schema.json        : JSON Schema for the Shared Symbolic Substrate (SSS).
- sss_ref.py             : SSS builder/validator and NBHS-512 *stub* sealer (uses SHA-512 as placeholder).
- morphspec_runner.py    : Tiny morphogenesis-like simulator producing a numeric field signature.
- demo_run.py            : End-to-end: build example SSS, run morph, emit session_v1.ostph.json (bundle).
- pickup.morph.json      : Example MorphSpec in JSON form.

Quick start
-----------
1) Validate & write an example SSS:
   python sss_ref.py --schema sss_schema.json --out sss_example.json

2) Run morph runner:
   python morphspec_runner.py --morph pickup.morph.json --seed osmtriph --out morph_result.json

3) Build a bundle:
   python demo_run.py

Outputs
-------
- sss_example.json
- morph_result.json
- session_v1.ostph.json  (contains SSS + morph result + NBHS-512 stub seal)

Notes
-----
- NBHS-512 here is a stub for demonstration (backed by SHA-512). Replace with the true NBHS-512 when ready.
- The morph simulator is intentionally lightweight and deterministic for reproducible demos.
