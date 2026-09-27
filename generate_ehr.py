import os
import csv
import re
import time
import random
import pandas as pd
from typing import List, Literal
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types as genai_types

# ---------------------------------------------------------
# Load environment variables (GEMINI_API_KEY)
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file. Please add GEMINI_API_KEY=<your-key> to .env")

# ---------------------------------------------------------
# Initialise Gemini client (google.genai SDK)
# ---------------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)


class DailyQuotaExhausted(RuntimeError):
    """Raised when the API reports a per-day free-tier quota is exhausted.
    Retrying within the same day will not help; the caller should abort.
    """

# ---------------------------------------------------------
# Pydantic schema definitions (strict traceability)
# ---------------------------------------------------------
class EHREvent(BaseModel):
    relative_time: Literal["T-10 years", "T-8 years", "T-5 years", "T-3 years", "T-1 year", "T-6 months", "T-3 months", "T-1 month", "T-2 weeks"]
    event_type: Literal["diagnosis", "medication", "symptom", "investigation", "procedure", "followup", "allergy", "hospitalization", "other"]
    description: str
    diagnosis: str
    medication: str
    investigation: str
    source_basis: Literal["current_encounter", "temporal_expansion"]
    evidence_basis: str

class EHRHistory(BaseModel):
    encounter_id: str
    manual_review_required: bool
    events: List[EHREvent]

class StrictValidationResult(BaseModel):
    is_strictly_supported: bool
    rejection_reason: str

# ---------------------------------------------------------
# Configuration (paths, ordering)
# ---------------------------------------------------------
BASE_DIR = r"c:\\Users\\rajee\\Downloads\\clinguard-dataset"
ASSIGNMENT_FILE = os.path.join(BASE_DIR, "member3_assignment.csv")
CHALLENGE_DIR = os.path.join(BASE_DIR, "aci-bench-corpus", "challenge_data")
AUDIT_FILE = os.path.join(BASE_DIR, "member3_ehr_audit.csv")
FAIL_FILE = os.path.join(BASE_DIR, "failed_encounters.csv")

TIME_ORDER = {
    "T-10 years": 1,
    "T-8 years": 2,
    "T-5 years": 3,
    "T-3 years": 4,
    "T-1 year": 5,
    "T-6 months": 6,
    "T-3 months": 7,
    "T-1 month": 8,
    "T-2 weeks": 9,
}

# ---------------------------------------------------------
# EHR extraction system prompt (defined once, reused)
# ---------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = (
    "You are a strict clinical data extractor. "
    "Generate historical EHR events (prior to T-0) based ONLY on explicit facts in the text.\n"
    "RULES:\n"
    "1. Only directly supported historical facts. No clinical plausibility inference.\n"
    "2. No minimum quota. If zero historical events can be safely derived, "
    "return an empty events list and set manual_review_required to true.\n"
    "3. T-0 (current encounter/HPI) is excluded.\n"
    "4. Every populated field must be independently supported by evidence_basis. "
    "Do not infer a diagnosis merely because a medication is mentioned.\n"
    "5. No invented doses, routes, dates, results, diagnoses, or outcomes.\n"
    "6. Do not include generic \"previous visit\" events unless they contain meaningful clinical information."
)

# ---------------------------------------------------------
# Module-level Gemini API call wrapper with retry logic
# ---------------------------------------------------------
def call_genai(model_name: str, prompt: str, schema, *,
               system_instruction: str = "", context: str = "") -> object:
    """
    Call the Gemini API with structured JSON output via the google.genai SDK.

    - Retries on per-minute quota errors, honouring the API-provided retry_delay.
    - Raises DailyQuotaExhausted immediately when a per-day cap is hit
      (retrying within the same day is futile).
    - Falls back to exponential backoff with jitter when no API delay is given.
    """
    max_retries = 8
    base_backoff = 2  # seconds

    cfg_kwargs = dict(
        response_mime_type="application/json",
        response_schema=schema,
    )
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**cfg_kwargs),
            )
            return response
        except Exception as exc:
            msg = str(exc)
            is_quota_error = (
                "Quota exceeded" in msg
                or "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
            )
            if is_quota_error:
                # Fast-fail on daily quota: no retry will help today
                if "PerDay" in msg or "PerModelPerDay" in msg:
                    tag = f"[{context}] " if context else ""
                    print(
                        f"{tag}[!!] DAILY quota exhausted. "
                        "This resets at midnight Pacific Time. Aborting."
                    )
                    raise DailyQuotaExhausted(
                        f"Daily free-tier quota exhausted{f' for {context}' if context else ''}. "
                        "Retry tomorrow after the quota resets."
                    ) from exc

                # Per-minute quota: honour API-provided retry_delay
                delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                api_delay = float(delay_match.group(1)) if delay_match else None

                if api_delay is not None:
                    wait = api_delay
                else:
                    # Exponential backoff: 2, 4, 8, 16, 32, 64, 128, 256 s
                    wait = base_backoff * (2 ** (attempt - 1))

                # Add random jitter (0-10% of wait, capped at 5 s)
                jitter = random.uniform(0, min(wait * 0.10, 5.0))
                wait += jitter

                tag = f"[{context}] " if context else ""
                print(
                    f"{tag}[!] Quota error (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait:.1f}s before retry."
                )
                time.sleep(wait)
                continue
            else:
                raise  # non-quota errors bubble up immediately

    raise RuntimeError(
        f"Gemini API call failed after {max_retries} retries"
        + (f" for {context}" if context else "") + "."
    )

# ---------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------
def load_source_data(source_file: str, encounter_id: str):
    df = pd.read_csv(os.path.join(CHALLENGE_DIR, source_file))
    id_col = "encounter_id" if "encounter_id" in df.columns else "id"
    row = df[df[id_col] == encounter_id]
    if row.empty:
        return None, None
    return row.iloc[0]["dialogue"], row.iloc[0]["note"]

def is_encounter_completed(output_file: str, encounter_id: str) -> bool:
    if not os.path.exists(output_file):
        return False
    df = pd.read_csv(output_file)
    return f"P_{encounter_id}" in df["patient_id"].values

def log_failure(encounter_id: str, split: str, error_msg: str):
    file_exists = os.path.exists(FAIL_FILE)
    with open(FAIL_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["encounter_id", "split", "error"])
        writer.writerow([encounter_id, split, error_msg])

def save_audit_csv(encounter_id: str, manual_review: bool, events: List[EHREvent]):
    file_exists = os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "encounter_id",
                "relative_time",
                "event_type",
                "description",
                "diagnosis",
                "medication",
                "investigation",
                "source_basis",
                "evidence_basis",
                "manual_review_required",
            ])
        if not events:
            writer.writerow([encounter_id, "", "", "", "", "", "", "", "", manual_review])
        else:
            for ev in events:
                writer.writerow([
                    encounter_id,
                    ev.relative_time,
                    ev.event_type,
                    ev.description,
                    ev.diagnosis,
                    ev.medication,
                    ev.investigation,
                    ev.source_basis,
                    ev.evidence_basis,
                    manual_review,
                ])

def save_events_to_csv(encounter_id: str, events: List[EHREvent], output_file: str):
    if not events:
        return
    file_exists = os.path.exists(output_file)
    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "patient_id",
                "encounter_id",
                "relative_time",
                "event_type",
                "description",
                "diagnosis",
                "medication",
                "investigation",
                "source_basis",
            ])
        patient_id = f"P_{encounter_id}"
        for ev in events:
            writer.writerow([
                patient_id,
                encounter_id,
                ev.relative_time,
                ev.event_type,
                ev.description,
                ev.diagnosis,
                ev.medication,
                ev.investigation,
                ev.source_basis,
            ])

# ---------------------------------------------------------
# Hard validator (Merciless auditor)
# ---------------------------------------------------------
def run_hard_validator(event: EHREvent) -> bool:
    prompt = (
        "You are a merciless auditor enforcing absolute traceability.\n"
        "Rule: No non-empty output field may contain information that is not directly stated "
        "in the evidence_basis. No clinical inference allowed.\n\n"
        f"Evidence Basis: \"{event.evidence_basis}\"\n\n"
        "Generated Fields:\n"
        f"- description: {event.description}\n"
        f"- diagnosis: {event.diagnosis}\n"
        f"- medication: {event.medication}\n"
        f"- investigation: {event.investigation}\n\n"
        "If ANY generated field contains information, context, or links "
        "(e.g., 'started Imitrex for migraines') that is NOT explicitly stated "
        "in the Evidence Basis, you must reject it."
    )
    completion = call_genai(
        "gemini-3.1-flash-lite", prompt, StrictValidationResult,
        context=f"validator/{event.event_type}"
    )
    result = StrictValidationResult.model_validate_json(completion.text)
    if not result.is_strictly_supported:
        print(f"      [!] Validator REJECTED: {result.rejection_reason}")
    return result.is_strictly_supported

# ---------------------------------------------------------
# Shared encounter processor
# ---------------------------------------------------------
def _process_one(enc_id: str, split: str, source_file: str, output_file: str) -> bool:
    """
    Extract, validate, and immediately save one encounter.
    Returns True on success, False on failure.
    """
    if is_encounter_completed(output_file, enc_id):
        print(f"[{split}] Skipping {enc_id}: Already completed.")
        return True

    dialogue, note = load_source_data(source_file, enc_id)
    if not dialogue:
        log_failure(enc_id, split, "Source data not found")
        print(f"[{split}] MISSING source data for {enc_id}")
        return False

    try:
        user_prompt = (
            f"Encounter ID: {enc_id}\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            f"Clinical Note:\n{note}"
        )
        completion = call_genai(
            "gemini-3.1-pro-preview", user_prompt, EHRHistory,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            context=enc_id,
        )
        ehr_history = EHRHistory.model_validate_json(completion.text)

        # Temporal order validation (unchanged logic)
        last_val = 0
        validated_events: List[EHREvent] = []
        for ev in ehr_history.events:
            curr_val = TIME_ORDER[ev.relative_time]
            if curr_val < last_val:
                raise ValueError("Events are not strictly chronologically ordered.")
            last_val = curr_val
            if run_hard_validator(ev):
                validated_events.append(ev)

        manual_review = ehr_history.manual_review_required or not validated_events

        # Immediately persist before moving to next encounter
        save_audit_csv(enc_id, manual_review, validated_events)
        save_events_to_csv(enc_id, validated_events, output_file)
        print(f"[{split}] SUCCESS {enc_id} -> {len(validated_events)} events. "
              f"Manual review: {manual_review}")
        return True

    except DailyQuotaExhausted:
        raise  # propagate up so the main loop aborts immediately
    except Exception as exc:
        log_failure(enc_id, split, str(exc))
        print(f"[{split}] FAILED {enc_id}: {exc}")
        return False

# ---------------------------------------------------------
# API smoke-test: one encounter before full run
# ---------------------------------------------------------
def test_one_encounter() -> bool:
    """
    Run the first incomplete encounter as a quota-handling smoke test.
    Returns True if it succeeded (or everything is done), False otherwise.
    """
    print("=" * 60)
    print("[Test] Running one-encounter smoke test ...")
    df_assignment = pd.read_csv(ASSIGNMENT_FILE)


    for _, row in df_assignment.iterrows():
        enc_id = row["encounter_id"]
        split = row["split"]
        source_file = row["source_file"]
        output_file = os.path.join(BASE_DIR, row["output_file"])

        if is_encounter_completed(output_file, enc_id):
            continue  # find first incomplete

        ok = _process_one(enc_id, split, source_file, output_file)
        if ok:
            print(f"[Test] Smoke test PASSED for {enc_id}. Proceeding to full run.")
        else:
            print(f"[Test] Smoke test FAILED for {enc_id}. Aborting.")
        print("=" * 60)
        return ok

    print("[Test] All encounters already completed. Nothing to test.")
    print("=" * 60)
    return True

# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def process_encounters():
    df_assignment = pd.read_csv(ASSIGNMENT_FILE)

    total = len(df_assignment)
    for idx, (_, row) in enumerate(df_assignment.iterrows(), start=1):
        enc_id = row["encounter_id"]
        split = row["split"]
        source_file = row["source_file"]
        output_file = os.path.join(BASE_DIR, row["output_file"])
        print(f"[{idx}/{total}] Processing {enc_id} ({split}) ...")
        _process_one(enc_id, split, source_file, output_file)

if __name__ == "__main__":
    try:
        # Gate: run smoke test first; only proceed if it passes
        if test_one_encounter():
            process_encounters()
        else:
            print("Smoke test failed - full run aborted. Check quota status and retry.")
    except DailyQuotaExhausted as dqe:
        print(f"\n[ABORT] {dqe}")