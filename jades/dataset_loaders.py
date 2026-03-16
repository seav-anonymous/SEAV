"""
Dataset loading utilities for scripts (not part of core JADES logic).

Supported datasets:
- JailbreakQR (local JSON)
- GPTFuzz (local CSV + question_list.csv)
- JBB-Behaviors (HF: judge_comparison)
- labeling_subsets_subset25.xlsx (local XLSX with sheets and query/response columns)
- dishonesty mcq_evaluation paired A/C JSONL (local JSONL produced by sample_mcq_A_C_one_per_query.py)
"""

from __future__ import annotations

import csv
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DatasetRow:
    dataset: str
    row_id: str
    query: str
    response: str
    meta: Dict[str, Any]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jailbreakqr_json(path: Path) -> List[DatasetRow]:
    data = _load_json(path)
    examples: Iterable[Dict[str, Any]]
    if isinstance(data, dict):
        examples = data.get("examples", [])
        if not isinstance(examples, list):
            raise ValueError(f"Unexpected schema in {path}: dict without list under 'examples'")
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(f"Unexpected schema in {path}: {type(data)}")

    rows: List[DatasetRow] = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            continue
        row_id = str(ex.get("index", i))
        query = str(ex.get("goal") or ex.get("prompt") or "")
        response = str(ex.get("response") or "")
        rows.append(DatasetRow(dataset="jailbreakqr", row_id=row_id, query=query, response=response, meta=dict(ex)))
    return rows


def load_gptfuzz(
    *,
    all_labeled_csv: Path,
    question_list_csv: Path,
) -> List[DatasetRow]:
    question_map: Dict[str, str] = {}
    with question_list_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = (row.get("index") or "").strip()
            text = row.get("text") or ""
            if idx:
                question_map[idx] = text

    rows: List[DatasetRow] = []
    with all_labeled_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {all_labeled_csv}")
        if "question" not in reader.fieldnames or "Response" not in reader.fieldnames:
            raise ValueError(
                f"Unexpected GPTFuzz schema in {all_labeled_csv}; expected columns include 'question' and 'Response', got: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            q_idx = str(row.get("question") or "").strip()
            query = question_map.get(q_idx, q_idx)
            response = str(row.get("Response") or "")
            meta = dict(row)
            meta["question_text"] = query
            rows.append(DatasetRow(dataset="gptfuzz", row_id=str(i), query=query, response=response, meta=meta))
    return rows


def load_jbb_behaviors_judge_comparison() -> List[DatasetRow]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face `datasets` is required to load JBB-Behaviors judge_comparison. "
            "Run scripts with the `jades` conda env."
        ) from e

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")["test"]
    rows: List[DatasetRow] = []
    for i in range(len(ds)):
        ex = dict(ds[i])
        query = str(ex.get("goal") or "")
        response = str(ex.get("target_response") or "")
        rows.append(
            DatasetRow(
                dataset="jbb_behaviors",
                row_id=f"judge_comparison:test:{i}",
                query=query,
                response=response,
                meta=ex,
            )
        )
    return rows


def _xlsx_sheet_map(xlsx_path: Path) -> Dict[str, str]:
    """
    Return mapping of sheet name -> worksheet xml path inside the xlsx zip.
    Designed for the lightweight XLSX written by make_labeling_subsets.py
    (inlineStr cells, no sharedStrings).
    """
    with zipfile.ZipFile(xlsx_path) as z:
        wb_xml = z.read("xl/workbook.xml")
        rels_xml = z.read("xl/_rels/workbook.xml.rels")

    wb_root = ET.fromstring(wb_xml)
    rels_root = ET.fromstring(rels_xml)

    ns_wb = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rid_to_target = {
        rel.attrib["Id"]: "xl/" + rel.attrib["Target"]
        for rel in rels_root.findall("r:Relationship", ns_rel)
        if rel.attrib.get("Target", "").startswith("worksheets/")
    }

    sheet_map: Dict[str, str] = {}
    for sheet in wb_root.findall("m:sheets/m:sheet", ns_wb):
        name = sheet.attrib.get("name")
        rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if name and rid and rid in rid_to_target:
            sheet_map[name] = rid_to_target[rid]
    return sheet_map


def _load_shared_strings(xlsx_path: Path) -> List[str]:
    """
    Load shared strings from a workbook if present.

    Some XLSX files (e.g., edited in Excel) store most strings in
    xl/sharedStrings.xml with cells referencing them via t="s".
    """
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            xml = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(xml)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: List[str] = []
    for si in root.findall("m:si", ns):
        parts = [t_el.text or "" for t_el in si.findall(".//m:t", ns)]
        strings.append("".join(parts))
    return strings


def _cell_text(cell: ET.Element, *, shared_strings: Optional[List[str]] = None) -> str:
    t = cell.attrib.get("t")
    if t == "s":
        v = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
        if v is None or v.text is None:
            return ""
        try:
            idx = int(v.text)
        except ValueError:
            return v.text
        if shared_strings and 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return v.text
    if t == "inlineStr":
        texts = [
            t_el.text or ""
            for t_el in cell.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        ]
        return "".join(texts)
    v = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
    return (v.text or "") if v is not None else ""


def load_labeling_subsets_xlsx(
    xlsx_path: Path,
    *,
    subset: str = "all",
    first_n_per_sheet: Optional[int] = None,
    include_all_columns: bool = False,
) -> List[DatasetRow]:
    """
    Load rows from labeling_subsets_subset25.xlsx.
    Expected columns: query, response (first row header).
    Returns DatasetRow with dataset=sheet_name, row_id=sheet:row.
    """
    sheet_map = _xlsx_sheet_map(xlsx_path)
    if subset != "all" and subset not in sheet_map:
        raise ValueError(f"Unknown subset/sheet '{subset}'. Available: {sorted(sheet_map)}")

    to_load = sheet_map.items() if subset == "all" else [(subset, sheet_map[subset])]
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared_strings = _load_shared_strings(xlsx_path)

    rows_out: List[DatasetRow] = []
    with zipfile.ZipFile(xlsx_path) as z:
        for sheet_name, sheet_path in to_load:
            sheet_xml = z.read(sheet_path)
            root = ET.fromstring(sheet_xml)
            sheet_data = root.find("m:sheetData", ns)
            if sheet_data is None:
                continue
            rows = sheet_data.findall("m:row", ns)
            if not rows:
                continue

            header_cells = rows[0].findall("m:c", ns)
            col_to_header: Dict[str, str] = {}
            for cell in header_cells:
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                col_to_header[col] = _cell_text(cell, shared_strings=shared_strings).strip()

            loaded_for_sheet = 0
            for row_el in rows[1:]:
                row_idx = row_el.attrib.get("r", "")
                cell_map: Dict[str, str] = {}
                for cell in row_el.findall("m:c", ns):
                    ref = cell.attrib.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    header = col_to_header.get(col, col)
                    cell_map[header] = _cell_text(cell, shared_strings=shared_strings)

                query = (cell_map.get("query") or "").strip()
                response = (cell_map.get("response") or "").strip()
                adv_prompt = (cell_map.get("adversarial prompts") or "").strip()
                if not response:
                    continue
                if not query and not adv_prompt:
                    continue
                effective_query = query or adv_prompt

                meta: Dict[str, Any] = {"subset": sheet_name, "row": row_idx}
                if include_all_columns:
                    meta["columns"] = dict(cell_map)
                rows_out.append(
                    DatasetRow(
                        dataset=sheet_name,
                        row_id=f"{sheet_name}:{row_idx}",
                        query=effective_query,
                        response=response,
                        meta=meta,
                    )
                )

                loaded_for_sheet += 1
                if first_n_per_sheet is not None and loaded_for_sheet >= first_n_per_sheet:
                    break
    return rows_out


def load_mcq_paired_jsonl(path: Path) -> List[DatasetRow]:
    """
    Load MCQ evaluation rows (A/C) from local JSONL.

    Supported input formats:

    1) Paired-per-query JSONL (produced by sample_mcq_A_C_one_per_query.py).
       Each line is:
         {"user_query": ..., "A": {...}, "C": {...}, ...}
       We emit TWO DatasetRow per line:
         - one for the "A" response
         - one for the "C" response

    2) Per-sample JSONL (e.g., filtered subsets derived from mcq_output_A.jsonl / mcq_output_C.jsonl).
       Each line is a single sample with fields like:
         {"user_query": ..., "mcq_output": "A"|"C", "draft_response": ..., "harmbench_judgment": ...}
       We emit ONE DatasetRow per line.

    Ground-truth label convention (stored in meta["human"]):
      - A: always False (NOT jailbroken)
      - C: derived from harmbench_judgment ("Yes" => True, "No" => False, else None)
    """
    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            user_query = str(obj.get("user_query") or "").strip()
            if not user_query:
                continue

            # Format 1: paired-per-query.
            if "A" in obj or "C" in obj:
                for choice in ("A", "C"):
                    rec = obj.get(choice)
                    if not isinstance(rec, dict):
                        continue

                    response = str(rec.get("draft_response") or rec.get("full_response") or "").strip()
                    if not response:
                        continue

                    meta = dict(rec)
                    meta["paired_choice"] = choice
                    meta["paired_final_label"] = obj.get("final_label")
                    meta["paired_count_in_A"] = obj.get("count_in_A")
                    meta["paired_count_in_C"] = obj.get("count_in_C")

                    if choice == "A":
                        meta["human"] = False
                        meta["human_source"] = "mcq_output==A"
                    else:
                        hb = str(rec.get("harmbench_judgment") or "").strip().lower()
                        if hb == "yes":
                            meta["human"] = True
                            meta["human_source"] = "harmbench_judgment==Yes"
                        elif hb == "no":
                            meta["human"] = False
                            meta["human_source"] = "harmbench_judgment==No"
                        else:
                            meta["human"] = None
                            meta["human_source"] = f"harmbench_judgment=={rec.get('harmbench_judgment')!r}"

                    rows.append(
                        DatasetRow(
                            dataset="mcq_paired",
                            row_id=f"{i}_{choice}",
                            query=user_query,
                            response=response,
                            meta=meta,
                        )
                    )
                continue

            # Format 2: per-sample.
            choice = str(obj.get("mcq_output") or "").strip().upper()
            if choice not in {"A", "C"}:
                continue

            response = str(obj.get("draft_response") or obj.get("full_response") or "").strip()
            if not response:
                continue

            meta = dict(obj)
            meta["paired_choice"] = choice

            if choice == "A":
                meta["human"] = False
                meta["human_source"] = "mcq_output==A"
            else:
                hb = str(obj.get("harmbench_judgment") or "").strip().lower()
                if hb == "yes":
                    meta["human"] = True
                    meta["human_source"] = "harmbench_judgment==Yes"
                elif hb == "no":
                    meta["human"] = False
                    meta["human_source"] = "harmbench_judgment==No"
                else:
                    meta["human"] = None
                    meta["human_source"] = f"harmbench_judgment=={obj.get('harmbench_judgment')!r}"

            # Prefer stable-ish IDs if present; otherwise fall back to the line index.
            eval_id = str(meta.get("evaluation_id") or meta.get("task_id") or i)
            row_num = str(meta.get("row_number") or i)
            rows.append(
                DatasetRow(
                    dataset="mcq_paired",
                    row_id=f"{eval_id}_{row_num}_{choice}",
                    query=user_query,
                    response=response,
                    meta=meta,
                )
            )

    return rows


def load_order_variants_jsonl(path: Path, variant: str) -> List[DatasetRow]:
    """
    Load order-variants dataset rows from local JSONL.

    Each line has: intent, original_response, wrong_order_response, alt_correct_response,
    plus structural metadata (n_steps, original_deps, violated_deps, etc.).

    Ground-truth label convention (stored in meta["human"]):
      - original: True (correct dependency order -> jailbroken)
      - alt_correct: True (alternative valid topological sort -> jailbroken)
      - wrong_order: False (violates dependency constraints -> non-jailbroken)
    """
    _VARIANT_FIELD = {
        "original": "original_response",
        "wrong_order": "wrong_order_response",
        "alt_correct": "alt_correct_response",
    }
    if variant not in _VARIANT_FIELD:
        raise ValueError(f"Unknown variant {variant!r}. Must be one of: {list(_VARIANT_FIELD)}")

    response_field = _VARIANT_FIELD[variant]
    human_label = variant in {"original", "alt_correct"}

    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            query = str(obj.get("intent") or "").strip()
            if not query:
                continue

            response = str(obj.get(response_field) or "").strip()
            if not response:
                print(f"[load_order_variants_jsonl] Warning: empty {response_field!r} at row {i}, skipping")
                continue

            meta = dict(obj)
            meta["human"] = human_label
            meta["variant"] = variant

            rows.append(
                DatasetRow(
                    dataset="order",
                    row_id=f"order:{variant}:{i}",
                    query=query,
                    response=response,
                    meta=meta,
                )
            )
    return rows
