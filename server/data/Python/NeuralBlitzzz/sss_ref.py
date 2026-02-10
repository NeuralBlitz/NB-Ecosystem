
import json, uuid, hashlib
from datetime import datetime

try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except Exception:
    HAVE_JSONSCHEMA = False

def nbhs512_stub(data_bytes: bytes) -> str:
    return hashlib.sha512(data_bytes).hexdigest()

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def new_uuid() -> str:
    return str(uuid.uuid4())

def validate_sss(doc: dict, schema: dict) -> None:
    if not HAVE_JSONSCHEMA:
        required = ["version","id","createdAt","ssType","payload"]
        for k in required:
            if k not in doc:
                raise ValueError(f"Missing required field: {k}")
        return
    jsonschema.validate(instance=doc, schema=schema)

def build_example_sss(created_by="demo@osmtrip") -> dict:
    ent_id = new_uuid()
    sym_id = new_uuid()
    trace_id = "trace-0002"
    ma_id = "m-anchor-0001"
    sss = {
      "version": "1.0.0",
      "id": new_uuid(),
      "createdAt": now_iso(),
      "createdBy": created_by,
      "ssType": "SSS",
      "payload": {
        "entities": [
          {
            "id": ent_id,
            "type":"Grasper",
            "label":"soft_gripper_v1",
            "attributes":{"material":"silicone","compliance":0.32},
            "temporal":{"validFrom": now_iso()},
            "provenanceRef":"trace-0001"
          }
        ],
        "symbols": [
          {
            "id": sym_id,
            "lang":"symp",
            "content":"Plan pick_and_place { steps:[approach(), conform(), lift()] }",
            "links":[ent_id],
            "metadata":{"complexity":0.12}
          }
        ],
        "morphAnchors": [
          {
            "id": ma_id,
            "anchorType":"3d",
            "params":{"geometry":"parametric_tip_v1","baseRadius":0.012},
            "attachedSymbol": sym_id,
            "provenanceRef": trace_id
          }
        ],
        "constraints": [
          {
            "id":"cons-0001",
            "expr":"Grasper.compliance >= 0.8 * Surface.softness",
            "level":"hard",
            "weight":1.0,
            "source":"ontology:grip.ontosp",
            "confidence":0.97
          }
        ],
        "processEdges": [
          {
            "id":"pe-0001",
            "from": sym_id,
            "to": ma_id,
            "transform":{"type":"attach","params":{}},
            "timestamp": now_iso(),
            "provenanceRef": trace_id
          }
        ],
        "traceUnits": [
          {
            "id": trace_id,
            "actor":"MorphoEngine::v0.2",
            "action":"morphogenesis_run",
            "inputs":[sym_id],
            "outputs":[ma_id],
            "details":{"simSteps":1200,"rdParameters":{"activator":1.0,"inhibitor":0.55}},
            "confidence":0.92,
            "timestamp": now_iso(),
            "signature":{"sig":"MEW...","scheme":"ed25519"}
          }
        ],
        "motifs": [
          {
            "id":"motif-0001",
            "summary":"gripTip:concave|radius=0.012|compliance=0.32",
            "privacy":{"level":"dp","dpParams":{"epsilon":0.5}},
            "linkBack": trace_id
          }
        ],
        "sessionMeta": {
          "intent":"soft_grip_prototype",
          "charterLock": True,
          "tags":["soft-robotics","prototype"]
        }
      }
    }
    return sss

def seal_document(doc: dict) -> str:
    data = json.dumps(doc, sort_keys=True).encode("utf-8")
    return nbhs512_stub(data)

if __name__ == "__main__":
    import argparse, sys, os
    ap = argparse.ArgumentParser(description="SSS reference tools")
    ap.add_argument("--schema", type=str, required=True, help="Path to sss_schema.json")
    ap.add_argument("--out", type=str, default="sss_example.json")
    args = ap.parse_args()

    with open(args.schema, "r") as f:
        schema = json.load(f)
    sss = build_example_sss()
    try:
        validate_sss(sss, schema)
    except Exception as e:
        print("Validation error:", e, file=sys.stderr)
        sys.exit(2)

    with open(args.out, "w") as f:
        json.dump(sss, f, indent=2)

    print("Wrote", args.out)
    print("NBHS-512 (stub, SHA-512) seal:", seal_document(sss))
