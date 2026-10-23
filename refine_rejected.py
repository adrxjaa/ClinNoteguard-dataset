import os
import csv
import json
import time
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

# Import schemas and helpers from generate_ehr
from generate_ehr import (
    EHREvent,
    StrictValidationResult,
    run_hard_validator,
    normalize_event,
    load_source_data,
    save_audit_csv,
    save_events_to_csv,
    save_diagnostics,
    BASE_DIR,
    CHALLENGE_DIR,
    MEMBER,
    ASSIGNMENT_FILE,
    AUDIT_FILE,
    STATUS_FILE,
    DIAGNOSTICS_FILE,
    REJECTED_FILE,
    EXTRACTION_MODEL,
    VALIDATOR_MODEL,
    call_genai,
    green,
    red,
    yellow,
    cyan,
    bold,
)

class RefinedEventResult(BaseModel):
    should_discard: bool
    discard_reason: str = ""
    refined_event: Optional[EHREvent] = None

REFINEMENT_SYSTEM_PROMPT = (
    "You are an expert clinical data auditor and precision editor.\n"
    "Your task is to fix an EHR candidate event that was REJECTED by a strict validator.\n\n"
    "COMMON REASONS FOR REJECTION AND HOW TO FIX THEM:\n"
    "1. 'Evidence basis too narrow / missing noun': Expand evidence_basis to include the complete exact sentence or dialogue turns containing the named condition, medication, lab test, or procedure.\n"
    "2. 'Unstated diagnosis or indication': If the evidence mentions a medication or test but does NOT explicitly state the diagnosis in that passage, leave diagnosis='' and do not say 'for [condition]' in description.\n"
    "3. 'Inferring clinical details/terms': Strip inferred terms (e.g. 'surgery', 'IV drugs', 'hypertension', anatomical side) if not literally stated in the evidence_basis.\n"
    "4. 'Non-historical / purely current visit (T-0)': If the event is solely about the current visit and has no valid past relative time, set should_discard=True.\n"
    "5. 'Cannot be grounded': If the candidate event is fundamentally hallucinated or cannot be strictly verified from the source text, set should_discard=True.\n\n"
    "RULES FOR refined_event:\n"
    "- relative_time MUST be one of: T-10 years, T-8 years, T-5 years, T-3 years, T-1 year, T-6 months, T-3 months, T-1 month, T-2 weeks\n"
    "- event_type MUST be one of: diagnosis, medication, symptom, investigation, procedure, followup, allergy, hospitalization, other\n"
    "- source_basis MUST be 'temporal_expansion' or 'current_encounter'\n"
    "- evidence_basis MUST be an EXACT substring from the dialogue or clinical note.\n"
    "- description, diagnosis, medication, investigation MUST contain ONLY facts explicitly present in the evidence_basis."
)

def refine_event(enc_id: str, split: str, dialogue: str, note: str, raw_event_dict: dict, rejection_reason: str) -> Optional[EHREvent]:
    prompt = (
        f"Encounter ID: {enc_id}\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        f"Clinical Note:\n{note}\n\n"
        f"--- PREVIOUS REJECTED EVENT ---\n"
        f"{json.dumps(raw_event_dict, indent=2)}\n\n"
        f"--- VALIDATOR REJECTION REASON ---\n"
        f"{rejection_reason}\n\n"
        "Please carefully analyze why this event failed validation. "
        "Find the exact source passage in the dialogue or note. "
        "If it can be strictly grounded, provide a corrected refined_event with an accurate, expanded evidence_basis and strictly grounded fields. "
        "Otherwise, set should_discard=True."
    )

    try:
        completion = call_genai(
            EXTRACTION_MODEL,
            prompt,
            RefinedEventResult,
            system_instruction=REFINEMENT_SYSTEM_PROMPT,
            context=f"refine/{enc_id}",
        )
        result = RefinedEventResult.model_validate_json(completion.text)
        if result.should_discard or not result.refined_event:
            print(yellow(f"      [discarded] {result.discard_reason or 'Not strictly groundable'}"))
            return None
        return result.refined_event
    except Exception as exc:
        print(red(f"      [error] Refinement call failed: {exc}"))
        return None


def get_existing_event_signatures(output_file: str) -> set:
    if not os.path.exists(output_file):
        return set()
    df = pd.read_csv(output_file)
    signatures = set()
    for _, row in df.iterrows():
        sig = (str(row.get("encounter_id")), str(row.get("relative_time")), str(row.get("description", "")).strip().lower())
        signatures.add(sig)
    return signatures


def run_refinement(dry_run: bool = False, target_encounters: Optional[list] = None):
    if not os.path.exists(REJECTED_FILE):
        print(red(f"No rejected file found at {REJECTED_FILE}"))
        return

    df_rej = pd.read_csv(REJECTED_FILE)
    df_assign = pd.read_csv(ASSIGNMENT_FILE).set_index("encounter_id")
    
    if target_encounters:
        df_rej = df_rej[df_rej["encounter_id"].isin(target_encounters)]

    print(cyan(bold(f"Starting refinement on {len(df_rej)} rejected events... (dry_run={dry_run})")))

    rescued_count = 0
    discarded_count = 0
    re_rejected_count = 0

    grouped = df_rej.groupby("encounter_id")
    for enc_id, group in grouped:
        if enc_id not in df_assign.index:
            continue
        
        row_assign = df_assign.loc[enc_id]
        if isinstance(row_assign, pd.DataFrame):
            row_assign = row_assign.iloc[0]
            
        split = row_assign["split"]
        source_file = row_assign["source_file"]
        output_file = os.path.join(BASE_DIR, row_assign["output_file"])

        dialogue, note = load_source_data(source_file, enc_id)
        if not dialogue:
            continue

        existing_sigs = get_existing_event_signatures(output_file)
        print(cyan(f"\n[{split}] Refining {enc_id} ({len(group)} candidate events) ..."))

        accepted_for_enc = []
        for _, rej_row in group.iterrows():
            raw_event = {
                "relative_time": rej_row["relative_time"],
                "event_type": rej_row["event_type"],
                "description": rej_row["description"],
                "diagnosis": str(rej_row.get("diagnosis", "") if pd.notna(rej_row.get("diagnosis")) else ""),
                "medication": str(rej_row.get("medication", "") if pd.notna(rej_row.get("medication")) else ""),
                "investigation": str(rej_row.get("investigation", "") if pd.notna(rej_row.get("investigation")) else ""),
                "source_basis": rej_row["source_basis"],
                "evidence_basis": str(rej_row.get("evidence_basis", "")),
            }
            reason = str(rej_row["rejection_reason"])

            print(yellow(f"  Candidate: {raw_event['description'][:70]}..."))
            refined_ev = refine_event(enc_id, split, dialogue, note, raw_event, reason)
            
            if not refined_ev:
                discarded_count += 1
                continue

            normalize_event(refined_ev)
            
            # Check duplicate signature
            sig = (enc_id, refined_ev.relative_time, refined_ev.description.strip().lower())
            if sig in existing_sigs:
                print(yellow(f"      [dup] Event already exists in output dataset: {refined_ev.description[:60]}"))
                continue

            # Run strict hard validator
            val_res = run_hard_validator(refined_ev)
            if val_res.is_strictly_supported:
                print(green(f"      [✓ PASSED] Rescued event: {refined_ev.description}"))
                rescued_count += 1
                accepted_for_enc.append(refined_ev)
                existing_sigs.add(sig)
            else:
                print(red(f"      [✗ RE-REJECTED] {val_res.rejection_reason}"))
                re_rejected_count += 1

        if accepted_for_enc and not dry_run:
            save_events_to_csv(enc_id, accepted_for_enc, output_file)
            save_audit_csv(enc_id, False, accepted_for_enc)
            # Update status
            df_status = pd.read_csv(STATUS_FILE)
            mask = df_status["encounter_id"] == enc_id
            if mask.any():
                df_status.loc[mask, "status"] = "completed"
                df_status.loc[mask, "manual_review_required"] = False
                df_status.loc[mask, "accepted_event_count"] += len(accepted_for_enc)
                df_status.to_csv(STATUS_FILE, index=False)

    print(cyan(bold("\n" + "=" * 60)))
    print(cyan(bold(f"Refinement Summary:")))
    print(green(f"  ✓ Rescued and validated events : {rescued_count}"))
    print(yellow(f"  - Truly invalid / discarded   : {discarded_count}"))
    print(red(f"  ✗ Still rejected by validator : {re_rejected_count}"))
    print(cyan(bold("=" * 60)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refine and recover rejected EHR events using LLM feedback loop.")
    parser.add_argument("--dry-run", action="store_true", help="Test refinement without modifying output CSVs.")
    parser.add_argument("--encounters", nargs="+", help="Specific encounter IDs to refine (e.g. D2N178 D2N001)")
    args = parser.parse_args()

    run_refinement(dry_run=args.dry_run, target_encounters=args.encounters)
