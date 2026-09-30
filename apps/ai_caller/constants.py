PATIENT_EXCEL_COLUMNS = (
    "name",
    "address",
    "dob",
    "doctor",
    "country_code",
    "phone_number",
    "live_agent_country_code",
    "live_agent_number",
)

PATIENT_UPLOAD_CONTENT_TYPES = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}

ALLOWED_UPLOAD_EXTENSIONS = set(PATIENT_UPLOAD_CONTENT_TYPES)
