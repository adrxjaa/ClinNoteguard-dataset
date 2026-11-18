"""
Grounded Verbalization of Synthea Longitudinal Timelines via Gemini API.

Reads:
  ClinNoteguard-dataset/synthea_timelines.jsonl
Outputs:
  ClinNoteguard-dataset/synthea_longitudinal_ehr.jsonl

Each encounter is converted into a structured JSON object containing:
  - patient_id, encounter_id, encounter_index, timestamp, encounter_class
  - demographics, structured_slice (conditions, observations, medications, procedures)
  - prior_summary (rolling summary of encounters 1..t-1)
  - clinical_note (SOAP / ED / Inpatient Progress / Urgent Care / Wellness note)
  - dialogue ([doctor] / [patient] transcript grounded in the exact Synthea facts)
"""

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class VerbalizedEncounter(BaseModel):
    note_style: str = Field(description="Style of note, e.g. Outpatient SOAP Note, ED Note, Inpatient Progress Note")
    clinical_note: str = Field(description="Grounded clinical encounter note reflecting longitudinal continuity")
    dialogue: str = Field(description="Realistic [doctor] and [patient] dialogue grounded strictly in the encounter facts")


STYLE_GUIDANCE = {
    "ambulatory": "Write a concise, structured Outpatient SOAP Note (Subjective, Objective, Assessment, Plan).",
    "emergency": "Write a focused, fast-paced Emergency Department (ED) Note (Chief Complaint, HPI, Physical Exam & Vitals, ED Course/Workup, Disposition).",
    "inpatient": "Write an Inpatient Hospital Progress Note (Interval History, Vitals & Labs, Assessment & Active Problems, Hospital Plan).",
    "urgentcare": "Write a brief, problem-focused Urgent Care Encounter Note (CC, Focused HPI, Exam/Labs, Assessment & Plan).",
    "wellness": "Write a Preventive Wellness & Annual Health Maintenance Note (Preventive Review, Vitals/Screenings, Chronic Disease Maintenance, Counseling).",
}


def _build_prompt(patient_rec: Dict[str, Any], enc: Dict[str, Any]) -> str:
    demo = patient_rec["demographics"]
    enc_class = enc.get("encounter_class", "ambulatory").lower()
    style_instr = STYLE_GUIDANCE.get(
        enc_class,
        "Write a realistic Clinical Encounter Note appropriate for the setting.",
    )

    payload = {
        "patient_demographics": {
            "gender": demo.get("gender"),
            "birthdate": demo.get("birthdate"),
        },
        "current_encounter": {
            "encounter_id": enc["encounter_id"],
            "timestamp": enc["start"],
            "encounter_class": enc_class,
            "description": enc.get("description"),
            "reason_description": enc.get("reason_description"),
            "conditions": enc.get("conditions", []),
            "observations": enc.get("observations", []),
            "medications": enc.get("medications", []),
            "procedures": enc.get("procedures", []),
        },
        "rolling_prior_history_1_to_t_minus_1": enc.get("prior_summary", {}),
    }

    return f"""You are a clinical documentation specialist converting structured Synthea EHR records into realistic clinical text.

STYLE & SETTING INSTRUCTION:
{style_instr}
Vary phrasing naturally for a {enc_class} setting so the dataset does not overfit to one rigid template.

STRICT GROUNDING RULES (CRITICAL):
1. Stay STRICTLY faithful to the structured Synthea facts provided below. Do NOT invent any new diagnoses, lab values, medications, dosages, allergies, or procedures that are not in `current_encounter` or `rolling_prior_history_1_to_t_minus_1`.
2. Use `rolling_prior_history_1_to_t_minus_1` to reflect longitudinal continuity where relevant (e.g., ongoing chronic conditions, prior medication changes or stopped meds, lab trends, or prior procedures).
3. In `dialogue`, write 6 to 10 natural turns alternating `[doctor] ...` and `[patient] ...` discussing ONLY the current encounter's symptoms/reasons, vitals/labs, medications, and relevant prior history from the JSON below.
4. Keep both `clinical_note` and `dialogue` clinically realistic, concise, and free of placeholder brackets like [Name].

STRUCTURED SYNTHEA INPUT:
{json.dumps(payload, indent=2)}
"""


def _verbalize_one(
    client: genai.Client,
    model_name: str,
    patient_rec: Dict[str, Any],
    enc: Dict[str, Any],
    max_retries: int = 4,
) -> Dict[str, Any]:
    prompt = _build_prompt(patient_rec, enc)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VerbalizedEncounter,
                    temperature=0.3,
                ),
            )
            parsed = json.loads(resp.text)
            return {
                "patient_id": patient_rec["patient_id"],
                "encounter_id": enc["encounter_id"],
                "encounter_index": enc["encounter_index"],
                "timestamp": enc["start"],
                "stop": enc.get("stop", ""),
                "encounter_class": enc["encounter_class"],
                "demographics": patient_rec["demographics"],
                "description": enc.get("description", ""),
                "reason_description": enc.get("reason_description", ""),
                "structured_slice": {
                    "conditions": enc.get("conditions", []),
                    "observations": enc.get("observations", []),
                    "medications": enc.get("medications", []),
                    "procedures": enc.get("procedures", []),
                },
                "prior_summary": enc.get("prior_summary", {}),
                "note_style": parsed.get("note_style", enc["encounter_class"]),
                "clinical_note": parsed["clinical_note"],
                "dialogue": parsed["dialogue"],
                "model": model_name,
            }
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** (attempt + 1), 16))
    raise RuntimeError(f"Failed encounter {enc['encounter_id']} after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Verbalize Synthea longitudinal timelines with Gemini API.")
    parser.add_argument("--timelines", type=Path, default=Path(__file__).parent / "synthea_timelines.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthea_longitudinal_ehr.jsonl")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--max-patients", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env")

    client = genai.Client(api_key=api_key)

    # Load already verbalized encounters for resume support
    done_eids: Set[str] = set()
    if args.out.exists():
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_eids.add(rec["encounter_id"])

    tasks = []
    with open(args.timelines, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.max_patients:
                break
            if not line.strip():
                continue
            pat_rec = json.loads(line)
            for enc in pat_rec["encounters"]:
                if enc["encounter_id"] not in done_eids:
                    tasks.append((pat_rec, enc))

    print(f"[+] Already verbalized: {len(done_eids)} encounters | Remaining to verbalize: {len(tasks)}")
    if not tasks:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0

    with open(args.out, "a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(_verbalize_one, client, args.model, pat, enc): enc["encounter_id"]
                for pat, enc in tasks
            }
            for fut in concurrent.futures.as_completed(future_map):
                eid = future_map[fut]
                try:
                    result = fut.result()
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == len(tasks):
                        print(f"  Verbalized {completed}/{len(tasks)} encounters...")
                except Exception as e:
                    failed += 1
                    print(f"  [!] Error on {eid}: {e}")

    print(f"[+] Done. Completed: {completed}, Failed: {failed}, Output: {args.out}")


if __name__ == "__main__":
    main()
