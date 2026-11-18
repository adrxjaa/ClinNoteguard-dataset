"""
Reconstruct Chronological Patient Timelines from Synthea CSV Exports.

Reads:
  - patients.csv
  - encounters.csv
  - conditions.csv
  - observations.csv
  - medications.csv
  - procedures.csv

Groups all records by PATIENT UUID, sorts encounters chronologically by START,
filters to multi-visit patients (>= min_encounters) with diverse ENCOUNTERCLASS
types, and for each encounter t bundles its structured slice alongside a compact
rolling summary of encounters 1..t-1.

Outputs:
  ClinNoteguard-dataset/synthea_timelines.jsonl
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _parse_iso(ts: Any) -> Optional[datetime]:
    if pd.isna(ts) or not str(ts).strip():
        return None
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _clean_str(val: Any) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def build_timelines(
    csv_dir: Path,
    out_path: Path,
    min_encounters: int = 4,
    max_patients: int = 80,
    max_encounters_per_patient: int = 8,
) -> Dict[str, Any]:
    patients_df = pd.read_csv(csv_dir / "patients.csv", dtype=str)
    encounters_df = pd.read_csv(csv_dir / "encounters.csv", dtype=str)
    conditions_df = pd.read_csv(csv_dir / "conditions.csv", dtype=str)
    observations_df = pd.read_csv(csv_dir / "observations.csv", dtype=str)
    medications_df = pd.read_csv(csv_dir / "medications.csv", dtype=str)
    procedures_df = pd.read_csv(csv_dir / "procedures.csv", dtype=str)

    patients_map: Dict[str, Dict[str, str]] = {}
    for _, row in patients_df.iterrows():
        pid = _clean_str(row.get("Id"))
        if pid:
            patients_map[pid] = {
                "patient_id": pid,
                "birthdate": _clean_str(row.get("BIRTHDATE")),
                "deathdate": _clean_str(row.get("DEATHDATE")),
                "first_name": _clean_str(row.get("FIRST")),
                "last_name": _clean_str(row.get("LAST")),
                "gender": _clean_str(row.get("GENDER")),
                "race": _clean_str(row.get("RACE")),
                "city": _clean_str(row.get("CITY")),
                "state": _clean_str(row.get("STATE")),
            }

    # Index encounter-linked tables by (PATIENT, ENCOUNTER)
    cond_by_enc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for _, row in conditions_df.iterrows():
        eid = _clean_str(row.get("ENCOUNTER"))
        if eid:
            cond_by_enc[eid].append({
                "start": _clean_str(row.get("START")),
                "stop": _clean_str(row.get("STOP")),
                "code": _clean_str(row.get("CODE")),
                "description": _clean_str(row.get("DESCRIPTION")),
            })

    obs_by_enc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for _, row in observations_df.iterrows():
        eid = _clean_str(row.get("ENCOUNTER"))
        if eid:
            obs_by_enc[eid].append({
                "date": _clean_str(row.get("DATE")),
                "code": _clean_str(row.get("CODE")),
                "description": _clean_str(row.get("DESCRIPTION")),
                "value": _clean_str(row.get("VALUE")),
                "units": _clean_str(row.get("UNITS")),
                "type": _clean_str(row.get("TYPE")),
            })

    med_by_enc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for _, row in medications_df.iterrows():
        eid = _clean_str(row.get("ENCOUNTER"))
        if eid:
            med_by_enc[eid].append({
                "start": _clean_str(row.get("START")),
                "stop": _clean_str(row.get("STOP")),
                "code": _clean_str(row.get("CODE")),
                "description": _clean_str(row.get("DESCRIPTION")),
                "reason_description": _clean_str(row.get("REASONDESCRIPTION")),
            })

    proc_by_enc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for _, row in procedures_df.iterrows():
        eid = _clean_str(row.get("ENCOUNTER"))
        if eid:
            proc_by_enc[eid].append({
                "date": _clean_str(row.get("START") or row.get("DATE")),
                "code": _clean_str(row.get("CODE")),
                "description": _clean_str(row.get("DESCRIPTION")),
                "reason_description": _clean_str(row.get("REASONDESCRIPTION")),
            })

    # Group encounters by patient
    enc_by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in encounters_df.iterrows():
        pid = _clean_str(row.get("PATIENT"))
        eid = _clean_str(row.get("Id"))
        start = _clean_str(row.get("START"))
        if not pid or not eid or not start:
            continue
        enc_by_patient[pid].append({
            "encounter_id": eid,
            "start": start,
            "stop": _clean_str(row.get("STOP")),
            "encounter_class": _clean_str(row.get("ENCOUNTERCLASS")).lower() or "ambulatory",
            "code": _clean_str(row.get("CODE")),
            "description": _clean_str(row.get("DESCRIPTION")),
            "reason_description": _clean_str(row.get("REASONDESCRIPTION")),
            "_sort_dt": _parse_iso(start) or datetime.min,
        })

    # Score patients by encounter class diversity and clinical richness
    candidate_patients = []
    for pid, encs in enc_by_patient.items():
        if len(encs) < min_encounters or pid not in patients_map:
            continue
        encs.sort(key=lambda x: x["_sort_dt"])
        # Keep recent rich window up to max_encounters_per_patient
        # Prefer encounters that have at least one condition, observation, medication, or procedure
        rich_encs = [
            e for e in encs
            if (cond_by_enc[e["encounter_id"]] or obs_by_enc[e["encounter_id"]]
                or med_by_enc[e["encounter_id"]] or proc_by_enc[e["encounter_id"]])
        ]
        if len(rich_encs) < min_encounters:
            continue
        classes = {e["encounter_class"] for e in rich_encs}
        candidate_patients.append((len(classes), len(rich_encs), pid, rich_encs))

    # Sort descending by class diversity then encounter count
    candidate_patients.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = candidate_patients[:max_patients]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_encounters = 0
    class_counts: Dict[str, int] = defaultdict(int)

    with open(out_path, "w", encoding="utf-8") as fout:
        for _, _, pid, rich_encs in selected:
            # Take up to max_encounters_per_patient evenly spaced or most recent encounters
            if len(rich_encs) > max_encounters_per_patient:
                rich_encs = rich_encs[-max_encounters_per_patient:]

            pat_info = patients_map[pid]
            active_conditions: Dict[str, str] = {}
            resolved_conditions: Dict[str, str] = {}
            active_meds: Dict[str, Dict[str, str]] = {}
            stopped_meds: Dict[str, Dict[str, str]] = {}
            prior_labs: Dict[str, Dict[str, str]] = {}
            prior_procedures: List[Dict[str, str]] = []
            prior_visits_brief: List[str] = []

            timeline_encounters = []
            for idx, enc in enumerate(rich_encs):
                eid = enc["encounter_id"]
                enc_start = enc["start"]
                enc_dt = enc["_sort_dt"]

                # Update stopped meds/conditions whose stop date <= current encounter start
                for m_desc, m_info in list(active_meds.items()):
                    stop_dt = _parse_iso(m_info.get("stop"))
                    if stop_dt is not None and enc_dt != datetime.min and stop_dt <= enc_dt:
                        stopped_meds[m_desc] = m_info
                        active_meds.pop(m_desc, None)

                for c_desc, c_start in list(active_conditions.items()):
                    pass  # stop date handled below when condition had a stop date

                prior_summary = {
                    "prior_encounter_count": idx,
                    "prior_visits": list(prior_visits_brief[-4:]),
                    "ongoing_conditions": sorted(active_conditions.keys())[:10],
                    "resolved_conditions": sorted(resolved_conditions.keys())[:6],
                    "active_medications": [
                        f"{k} (since {v['start'][:10]})" for k, v in list(active_meds.items())[:10]
                    ],
                    "stopped_medications": [
                        f"{k} (stopped {v['stop'][:10]})" for k, v in list(stopped_meds.items())[:6]
                    ],
                    "recent_observations": [
                        f"{k}: {v['value']} {v['units']} ({v['date'][:10]})".strip()
                        for k, v in list(prior_labs.items())[:8]
                    ],
                    "prior_procedures": [
                        f"{p['description']} ({p['date'][:10]})" for p in prior_procedures[-5:]
                    ],
                }

                curr_conds = cond_by_enc.get(eid, [])
                curr_obs = obs_by_enc.get(eid, [])[:15]
                curr_meds = med_by_enc.get(eid, [])
                curr_procs = proc_by_enc.get(eid, [])

                timeline_encounters.append({
                    "encounter_index": idx,
                    "encounter_id": eid,
                    "start": enc["start"],
                    "stop": enc["stop"],
                    "encounter_class": enc["encounter_class"],
                    "code": enc["code"],
                    "description": enc["description"],
                    "reason_description": enc["reason_description"],
                    "conditions": curr_conds,
                    "observations": curr_obs,
                    "medications": curr_meds,
                    "procedures": curr_procs,
                    "prior_summary": prior_summary,
                })

                total_encounters += 1
                class_counts[enc["encounter_class"]] += 1

                # Roll forward state into 1..t summary for t+1
                for c in curr_conds:
                    desc = c["description"]
                    if not desc:
                        continue
                    if c["stop"]:
                        resolved_conditions[desc] = c["stop"]
                        active_conditions.pop(desc, None)
                    else:
                        active_conditions[desc] = c["start"]

                for m in curr_meds:
                    desc = m["description"]
                    if not desc:
                        continue
                    stop_dt = _parse_iso(m["stop"])
                    if m["stop"] and stop_dt is not None and stop_dt <= enc_dt:
                        stopped_meds[desc] = m
                        active_meds.pop(desc, None)
                    else:
                        active_meds[desc] = m

                for o in curr_obs:
                    desc = o["description"]
                    if desc and o["value"]:
                        prior_labs[desc] = o

                for p in curr_procs:
                    if p["description"]:
                        prior_procedures.append(p)

                reason_str = f" for {enc['reason_description']}" if enc["reason_description"] else ""
                prior_visits_brief.append(
                    f"{enc_start[:10]} [{enc['encounter_class']}]: {enc['description']}{reason_str}"
                )

            record = {
                "patient_id": pid,
                "demographics": pat_info,
                "num_encounters": len(timeline_encounters),
                "encounter_classes": sorted({e["encounter_class"] for e in timeline_encounters}),
                "encounters": timeline_encounters,
            }
            fout.write(json.dumps(record) + "\n")

    summary = {
        "patients_selected": len(selected),
        "total_encounters": total_encounters,
        "encounter_class_distribution": dict(class_counts),
        "output_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build longitudinal patient timelines from Synthea CSVs.")
    parser.add_argument("--csv-dir", type=Path, default=Path(__file__).parent / "synthea" / "output" / "csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthea_timelines.jsonl")
    parser.add_argument("--min-encounters", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=80)
    parser.add_argument("--max-encounters-per-patient", type=int, default=8)
    args = parser.parse_args()

    build_timelines(
        csv_dir=args.csv_dir,
        out_path=args.out,
        min_encounters=args.min_encounters,
        max_patients=args.max_patients,
        max_encounters_per_patient=args.max_encounters_per_patient,
    )
