import csv
import io
from pathlib import Path

from openpyxl import Workbook, load_workbook

from apps.ai_caller.constants import PATIENT_EXCEL_COLUMNS

HEADER_ALIASES = {
    "first_name": "first_name",
    "firstname": "first_name",
    "last_name": "last_name",
    "lastname": "last_name",
    "name": "name",
    "patient_name": "name",
    "address": "address",
    "dob": "dob",
    "date_of_birth": "dob",
    "doctor": "doctor",
    "service_name": "service_name",
    "service": "service_name",
    "country_code": "country_code",
    "phone": "phone_number",
    "phone_number": "phone_number",
}


def _normalize_header(value):
    return str(value or "").strip().lower().replace(" ", "_")


def build_patient_template_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Patients"
    sheet.append(list(PATIENT_EXCEL_COLUMNS))
    sheet.append(
        [
            "Maria",
            "Santos",
            "14 Oak Street, Dover, DE 19901",
            "1984-03-12",
            "Dr. Alan Brooks",
            "annual wellness visit",
            "",
            "3025550101",
        ]
    )
    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _map_headers(raw_headers):
    mapping = {}
    for index, header in enumerate(raw_headers):
        key = HEADER_ALIASES.get(_normalize_header(header))
        if key and key not in mapping:
            mapping[key] = index
    return mapping


def _row_to_dict(values, mapping):
    # Include legacy "name" so services can split it when first/last are blank.
    fields = list(PATIENT_EXCEL_COLUMNS) + ["name"]
    row = {}
    for field in fields:
        index = mapping.get(field)
        if index is None or index >= len(values):
            row[field] = ""
            continue
        value = values[index]
        row[field] = "" if value is None else str(value).strip()
    return row


def _iter_xlsx_rows(file_bytes):
    workbook = load_workbook(filename=io.BytesIO(file_bytes), data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []

    mapping = _map_headers(rows[0])
    parsed = []
    for row_number, values in enumerate(rows[1:], start=2):
        if values is None or all(cell is None or str(cell).strip() == "" for cell in values):
            continue
        parsed.append((row_number, _row_to_dict(list(values), mapping)))
    return parsed


def _iter_csv_rows(file_bytes):
    text = file_bytes.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []

    mapping = _map_headers(rows[0])
    parsed = []
    for row_number, values in enumerate(rows[1:], start=2):
        if not values or all(str(cell).strip() == "" for cell in values):
            continue
        parsed.append((row_number, _row_to_dict(values, mapping)))
    return parsed


def parse_patient_upload(file_bytes, filename):
    extension = Path(filename).suffix.lower()
    if extension == ".xlsx":
        return _iter_xlsx_rows(file_bytes)
    if extension == ".csv":
        return _iter_csv_rows(file_bytes)
    raise ValueError("Unsupported file type. Use .xlsx or .csv.")
