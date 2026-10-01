PATIENT_EXCEL_COLUMNS = (
    "first_name",
    "last_name",
    "address",
    "dob",
    "doctor",
    "service_name",
    "country_code",
    "phone_number",
)

PATIENT_UPLOAD_CONTENT_TYPES = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}

ALLOWED_UPLOAD_EXTENSIONS = set(PATIENT_UPLOAD_CONTENT_TYPES)
