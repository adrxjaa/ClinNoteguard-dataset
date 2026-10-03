import os
import csv
import re
import time
import random
import json
from datetime import datetime, timezone
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
# Terminal colour helpers (auto-disabled when not a TTY)
# ---------------------------------------------------------
import sys as _sys
_COLOUR = _sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

def green(t):   return _c("92", t)   # success, accepted events, info
def red(t):     return _c("91", t)   # errors, rejections, failures, aborts
def yellow(t):  return _c("93", t)   # warnings, retries, skips, quota
def cyan(t):    return _c("96", t)   # progress headers, model info
def bold(t):    return _c("1",  t)   # emphasis

# ---------------------------------------------------------
# Load environment variables (GEMINI_API_KEY)
# ---------------------------------------------------------
load_dotenv(override=True)
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
# ---------------------------------------------------------
# Universal config — change MEMBER to 1, 2, or 3 to switch assignment
# ---------------------------------------------------------
MEMBER = 1  # <-- set this to the member number you are processing

# BASE_DIR is always the folder that contains this script (works on any OS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.join(BASE_DIR, "aci-bench-corpus", "challenge_data")

ASSIGNMENT_FILE  = os.path.join(BASE_DIR, f"member{MEMBER}_assignment.csv")
AUDIT_FILE       = os.path.join(BASE_DIR, f"member{MEMBER}_ehr_audit.csv")
FAIL_FILE        = os.path.join(BASE_DIR, f"member{MEMBER}_failed_encounters.csv")
STATUS_FILE      = os.path.join(BASE_DIR, f"member{MEMBER}_processing_status.csv")
DIAGNOSTICS_FILE = os.path.join(BASE_DIR, f"member{MEMBER}_generation_diagnostics.jsonl")
REJECTED_FILE    = os.path.join(BASE_DIR, f"member{MEMBER}_rejected_events.csv")

# ---------------------------------------------------------
# Model selection
#   EXTRACTION_MODEL  — used for the main EHR extraction pass
#                       use the most capable available model for best data quality
#   VALIDATOR_MODEL   — used for the hard validator (binary yes/no),
#                       a lighter model is fine here and saves quota
# ---------------------------------------------------------
EXTRACTION_MODEL = "gemini-3.1-flash-lite" # high free-tier quota — can process all 69 encounters
VALIDATOR_MODEL  = "gemini-3.1-flash-lite" # same model, binary yes/no validation
# NOTE: gemini-3.5-flash has better quality but tiny free-tier daily quota (hits limit ~3 encounters)
# Switch EXTRACTION_MODEL to "gemini-3.5-flash" only if you have a paid API key

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
    "You must aggressively scan BOTH the dialogue and clinical note to perform EXHAUSTIVE EXTRACTION of explicitly stated historical clinical facts.\n"
    "Identify every explicitly stated historical event that can safely become a longitudinal EHR event. "
    "Look specifically for:\n"
    "- previous diagnoses or history of conditions\n"
    "- previous symptoms/conditions when explicitly stated as history\n"
    "- previous medications and medication starts/stops when explicitly stated\n"
    "- previous surgeries/procedures\n"
    "- previous investigations/tests when explicitly stated\n"
    "- previous hospitalizations\n"
    "- previous allergies\n"
    "- previous follow-up/history events when clinically meaningful\n"
    "Look for explicit temporal phrases such as: years ago, months ago, weeks ago, previously, prior, history of, past medical history, last visit, previously underwent, had [procedure] in [past time].\n\n"
    "RULES:\n"
    "1. The event must represent exactly what the evidence says, not what it implies. Clinically plausible information MUST NOT be inferred.\n"
    "2. If the source says a patient had a condition in the past, don't automatically turn it into a current diagnosis field entry.\n"
    "3. If the source mentions a medication and separately mentions a condition, don't connect them unless explicitly connected in the text.\n"
    "4. If the source gives a numerical value but doesn't identify the test, don't assign a test name.\n"
    "5. Don't turn treatment of a condition into an explicit diagnosis unless the source states the diagnosis.\n"
    "6. Don't add anatomical/location details that aren't in the evidence.\n"
    "7. Don't add phrases such as 'for depression', 'for back pain', 'for birth control', etc. unless explicitly stated.\n"
    "8. The evidence_basis should be the smallest exact source passage that supports the event.\n"
    "9. Every populated output field must be directly supported by that evidence basis.\n"
    "10. Explicit historical information MUST be extracted. T-0 (current encounter/HPI) is excluded. Convert to relative time based on the explicit temporal phrase.\n"
    "11. If an encounter truly contains no qualifying historical clinical event, return: 'events': []\n"
    "12. Events do NOT need to be produced in chronological order.\n\n"
    "VALID relative_time VALUES (you must use ONLY these exact strings — no others are accepted):\n"
    "  T-10 years, T-8 years, T-5 years, T-3 years, T-1 year,\n"
    "  T-6 months, T-3 months, T-1 month, T-2 weeks\n"
    "When the source temporal phrase does not exactly match a bucket, pick the NEAREST one.\n"
    "Examples: '2 years ago' → T-3 years | '6 years ago' → T-5 years | '18 months ago' → T-1 year | '4 weeks ago' → T-1 month\n\n"
    "EXAMPLES:\n"
    "Source: 'I had a lumbar fusion about six years ago.'\n"
    "Output: { 'relative_time': 'T-5 years', 'event_type': 'procedure', 'description': 'Patient had a lumbar fusion.', 'diagnosis': '', 'medication': '', 'investigation': '', 'source_basis': 'temporal_expansion', 'evidence_basis': 'had a lumbar fusion about six years ago' }\n\n"
    "Source: 'I had kidney stones about two years ago.'\n"
    "Output: { 'relative_time': 'T-3 years', 'event_type': 'diagnosis', 'description': 'Patient had kidney stones.', 'diagnosis': 'kidney stones', 'medication': '', 'investigation': '', 'source_basis': 'temporal_expansion', 'evidence_basis': 'had kidney stones about two years ago' }\n"
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
            is_transient_error = (
                "503" in msg
                or "UNAVAILABLE" in msg
                or "currently experiencing high demand" in msg
            )
            if is_quota_error:
                # Fast-fail on daily quota: no retry will help today
                if "PerDay" in msg or "PerModelPerDay" in msg:
                    tag = f"[{context}] " if context else ""
                    print(red(
                        f"{tag}[!!] DAILY quota exhausted. "
                        "This resets at midnight Pacific Time. Aborting."
                    ))
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
                print(yellow(
                    f"{tag}[!] Quota error (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait:.1f}s before retry."
                ))
                time.sleep(wait)
                continue
            elif is_transient_error:
                # 503 / model overloaded — back off and retry
                wait = base_backoff * (2 ** (attempt - 1))
                jitter = random.uniform(0, min(wait * 0.10, 5.0))
                wait += jitter
                tag = f"[{context}] " if context else ""
                print(yellow(
                    f"{tag}[!] Model unavailable / 503 (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait:.1f}s before retry."
                ))
                time.sleep(wait)
                continue
            else:
                raise  # non-quota, non-transient errors bubble up immediately

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

def is_encounter_completed(encounter_id: str) -> bool:
    if not os.path.exists(STATUS_FILE):
        return False
    df = pd.read_csv(STATUS_FILE)
    matches = df[(df["encounter_id"] == encounter_id) & (df["status"] == "completed")]
    return not matches.empty

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
def run_hard_validator(event: EHREvent) -> StrictValidationResult:
    prompt = (
        "You are a merciless auditor enforcing absolute traceability.\n"
        "Rule: No non-empty output field may contain information that is not directly stated "
        "in the evidence_basis. No clinical inference allowed.\n"
        "Exception: Standard medical terminology or synonymous phrasing that accurately represents the evidence is acceptable "
        "(e.g., 'appendectomy' for 'appendix out', 'treated with' for 'managed with'). Do not penalize minor grammatical or synonym-based rewording if no new clinical facts are inferred.\n\n"
        f"Evidence Basis: \"{event.evidence_basis}\"\n\n"
        "Generated Fields:\n"
        f"- description: {event.description}\n"
        f"- diagnosis: {event.diagnosis}\n"
        f"- medication: {event.medication}\n"
        f"- investigation: {event.investigation}\n\n"
        "If ANY generated field contains clinical information, context, or links "
        "(e.g., 'started Imitrex for migraines') that is NOT explicitly stated "
        "in the Evidence Basis, you must reject it."
    )
    completion = call_genai(
        VALIDATOR_MODEL, prompt, StrictValidationResult,
        context=f"validator/{event.event_type}"
    )
    result = StrictValidationResult.model_validate_json(completion.text)
    if not result.is_strictly_supported:
        print(red(f"      [✗] Validator REJECTED: {result.rejection_reason}"))
    return result


def normalize_event(ev: EHREvent):
    placeholders = {
        "none", "n/a", "na", "not applicable", "not available",
        "null", "not specified", "unspecified", "unknown", "n/a.",
        "-", "--", "N/A", "none.",
    }
    def _clean(val: str) -> str:
        return "" if val and val.strip().lower() in placeholders else val
    ev.description   = _clean(ev.description)
    ev.diagnosis     = _clean(ev.diagnosis)
    ev.medication    = _clean(ev.medication)
    ev.investigation = _clean(ev.investigation)

def save_status(split: str, encounter_id: str, status: str, accepted_count: int, manual_review: bool, error: str):
    file_exists = os.path.exists(STATUS_FILE)
    attempt_count = 1
    if file_exists:
        df = pd.read_csv(STATUS_FILE)
        attempt_count = len(df[df["encounter_id"] == encounter_id]) + 1
    with open(STATUS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["split", "encounter_id", "status", "accepted_event_count", "manual_review_required", "error", "attempt_count"])
        writer.writerow([split, encounter_id, status, accepted_count, manual_review, error, attempt_count])

def save_rejected_events(encounter_id: str, split: str, rejected_list: list):
    if not rejected_list:
        return
    file_exists = os.path.exists(REJECTED_FILE)
    with open(REJECTED_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["encounter_id", "split", "relative_time", "event_type", "description", "diagnosis", "medication", "investigation", "source_basis", "evidence_basis", "rejected_fields", "rejection_reason"])
        for ev, reason in rejected_list:
            writer.writerow([encounter_id, split, ev.relative_time, ev.event_type, ev.description, ev.diagnosis, ev.medication, ev.investigation, ev.source_basis, ev.evidence_basis, "", reason])

def save_diagnostics(data: dict):
    with open(DIAGNOSTICS_FILE, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

# ---------------------------------------------------------
# Shared encounter processor
# ---------------------------------------------------------
def _process_one(enc_id: str, split: str, source_file: str, output_file: str) -> bool:
    if is_encounter_completed(enc_id):
        print(cyan(f"[{split}] ⏭  Skipping {enc_id}: Already completed."))
        return True

    dialogue, note = load_source_data(source_file, enc_id)
    if not dialogue:
        save_status(split, enc_id, "failed", 0, False, "Source data not found")
        log_failure(enc_id, split, "Source data not found")
        print(red(f"[{split}] ✗ MISSING source data for {enc_id}"))
        return False

    try:
        user_prompt = f"Encounter ID: {enc_id}\n\nDialogue:\n{dialogue}\n\nClinical Note:\n{note}"
        completion = call_genai(
            EXTRACTION_MODEL, user_prompt, EHRHistory,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            context=enc_id,
        )
        print(cyan(f"  Model used: {EXTRACTION_MODEL}"))
        usage_dict = {}
        if hasattr(completion, 'usage_metadata') and completion.usage_metadata:
            try:
                usage_dict = completion.usage_metadata.model_dump()
            except Exception as _ue:
                usage_dict = str(completion.usage_metadata)
                print(yellow(f"  [warn] usage_metadata serialisation failed: {_ue}"))
            print(cyan(f"  Tokens — prompt: {getattr(completion.usage_metadata, 'prompt_token_count', '?')}  "
                       f"output: {getattr(completion.usage_metadata, 'candidates_token_count', '?')}  "
                       f"thinking: {getattr(completion.usage_metadata, 'thoughts_token_count', '?')}  "
                       f"total: {getattr(completion.usage_metadata, 'total_token_count', '?')}"))
        
        ehr_history = EHRHistory.model_validate_json(completion.text)
        
        ehr_history.events.sort(key=lambda ev: TIME_ORDER[ev.relative_time])
        
        diagnostic_data = {
            "encounter_id": enc_id,
            "split": split,
            "model": EXTRACTION_MODEL,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "raw_generated_json": ehr_history.model_dump(),
            "usage_metadata": usage_dict,
            "validation_status": "",
            "accepted_events": [],
            "rejected_events": [],
            "rejection_reasons": [],
            "chronological_validation_result": ""
        }

        # Temporal order validation
        last_val = 0
        chrono_error = None
        try:
            for ev in ehr_history.events:
                curr_val = TIME_ORDER[ev.relative_time]
                if curr_val < last_val:
                    raise ValueError("Events are not strictly chronologically ordered.")
                last_val = curr_val
        except Exception as e:
            chrono_error = str(e)
            
        if chrono_error:
            diagnostic_data["chronological_validation_result"] = "FAILED"
            save_diagnostics(diagnostic_data)
            save_status(split, enc_id, "failed", 0, False, chrono_error)
            log_failure(enc_id, split, chrono_error)
            print(red(f"[{split}] ✗ FAILED {enc_id} (chrono): {chrono_error}"))
            return False
            
        diagnostic_data["chronological_validation_result"] = "PASSED"

        validated_events = []
        rejected_list = []
        seen_evidence: set = set()
        for ev in ehr_history.events:
            normalize_event(ev)
            # Deduplicate: skip events with identical evidence_basis already accepted
            evidence_key = ev.evidence_basis.strip().lower()
            if evidence_key and evidence_key in seen_evidence:
                print(yellow(f"      [dup] Skipping duplicate event: {ev.evidence_basis[:80]}"))
                continue
            val_result = run_hard_validator(ev)
            if val_result.is_strictly_supported:
                validated_events.append(ev)
                if evidence_key:
                    seen_evidence.add(evidence_key)
                diagnostic_data["accepted_events"].append(ev.model_dump())
            else:
                rejected_list.append((ev, val_result.rejection_reason))
                diagnostic_data["rejected_events"].append(ev.model_dump())
                diagnostic_data["rejection_reasons"].append(val_result.rejection_reason)

        manual_review = ehr_history.manual_review_required or not validated_events
        diagnostic_data["validation_status"] = "COMPLETED"
        
        save_diagnostics(diagnostic_data)
        save_rejected_events(enc_id, split, rejected_list)
        save_audit_csv(enc_id, manual_review, validated_events)
        save_events_to_csv(enc_id, validated_events, output_file)
        
        status = "completed" if validated_events else "manual_review"
        save_status(split, enc_id, status, len(validated_events), manual_review, "")
        
        _summary = f"[{split}] ✓ {enc_id} → {len(validated_events)} events accepted"
        _summary += f"  ⚑ manual review" if manual_review else ""
        print(green(_summary) if validated_events else yellow(_summary + "  (0 events — manual review)"))
        return True

    except DailyQuotaExhausted:
        raise  # propagate up so the main loop aborts immediately
    except Exception as exc:
        save_status(split, enc_id, "failed", 0, False, str(exc))
        log_failure(enc_id, split, str(exc))
        print(red(f"[{split}] ✗ FAILED {enc_id}: {exc}"))
        return False

# ---------------------------------------------------------
# API smoke-test: one encounter before full run
# ---------------------------------------------------------
def test_one_encounter() -> bool:
    """
    Run the first incomplete encounter as a quota-handling smoke test.
    Returns True if it succeeded (or everything is done), False otherwise.
    """
    print(cyan(bold("=" * 60)))
    print(cyan(bold("[Test] Running one-encounter smoke test ...")))
    df_assignment = pd.read_csv(ASSIGNMENT_FILE)


    for _, row in df_assignment.iterrows():
        enc_id = row["encounter_id"]
        split = row["split"]
        source_file = row["source_file"]
        output_file = os.path.join(BASE_DIR, row["output_file"])

        if is_encounter_completed(enc_id):
            continue  # find first incomplete

        ok = _process_one(enc_id, split, source_file, output_file)
        if ok:
            print(green(f"[Test] ✓ Smoke test PASSED for {enc_id}. Proceeding to full run."))
        else:
            print(red(f"[Test] ✗ Smoke test FAILED for {enc_id}. Aborting."))
        print(cyan("=" * 60))
        return ok

    print(green("[Test] ✓ All encounters already completed. Nothing to test."))
    print(cyan("=" * 60))
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
        print(cyan(f"\n[{idx}/{total}] Processing {enc_id} ({split}) ..."))
        _process_one(enc_id, split, source_file, output_file)

if __name__ == "__main__":
    try:
        # Gate: run smoke test first; only proceed if it passes
        if test_one_encounter():
            # print("Smoke test completed successfully. Stopping before remaining encounters as requested.")
            process_encounters()
        else:
            print(red("Smoke test failed - full run aborted. Check quota status and retry."))
    except DailyQuotaExhausted as dqe:
        print(red(bold(f"\n[ABORT] {dqe}")))