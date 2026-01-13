
import json, os, uuid
from datetime import datetime
from sss_ref import build_example_sss, validate_sss, seal_document
from morphspec_runner import run_morph

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def bundle_ostph(sss_doc: dict, morph_result: dict, created_by="demo@osmtrip"):
    bundle = {
      "version":"1.0.0",
      "id": str(uuid.uuid4()),
      "createdAt": now_iso(),
      "createdBy": created_by,
      "ssType":"OSTPH-BUNDLE",
      "payload": {
        "sss": sss_doc,
        "morphResult": morph_result
      }
    }
    bundle["nbhs512_seal_stub"] = seal_document(bundle)
    return bundle

if __name__ == "__main__":
    import json, os, sys
    schema_path = os.path.join(os.path.dirname(__file__), "sss_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    sss = build_example_sss()
    validate_sss(sss, schema)
    morph_spec = {
        "Tip": {"baseRadius": 0.012, "fillet": 0.0012, "material":"silicone"},
        "RD": {"activator":1.0, "inhibitor":0.55, "iterations": 128},
        "Anchors": {"SYM":"pick_and_place"}
    }
    morph_res = run_morph(morph_spec, seed="osmtriph-demo")
    bundle = bundle_ostph(sss, morph_res)
    with open("session_v1.ostph.json","w") as f:
        json.dump(bundle, f, indent=2)
    print("Wrote session_v1.ostph.json")
