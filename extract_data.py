import json
from pathlib import Path
from pypdf import PdfReader

# Base path configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw" / "weather_forecasts"

def extract_pdf_text(pdf_path: Path) -> str:
    """Extracts text content from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def parse_wwis_json(json_path: Path) -> str:
    """Parses WWIS JSON into structured text for RAG ingestion."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    city_info = data.get("city", {})
    city_name = city_info.get("cityName", "Unknown")
    forecast_info = city_info.get("forecast", {})
    issue_date = forecast_info.get("issueDate", "N/A")
    forecast_days = forecast_info.get("forecastDay", [])
    
    formatted_doc = f"Source: WMO / PMD (WWIS)\nCity: {city_name}\nIssue Date: {issue_date}\n\nForecast Details:\n"
    for day in forecast_days:
        formatted_doc += (
            f"- Date: {day.get('forecastDate')}, Weather: {day.get('weather', 'N/A')}, "
            f"Min Temp: {day.get('minTemp')}°C, Max Temp: {day.get('maxTemp')}°C\n"
        )
    return formatted_doc

if __name__ == "__main__":
    # 1. Process PMD PDF
    pmd_folder = DATA_DIR / "pmd_daily_forecasts"
    pdf_files = list(pmd_folder.glob("*.pdf"))
    
    print("=== PMD PDF EXTRACTION ===")
    if pdf_files:
        pdf_text = extract_pdf_text(pdf_files[0])
        print(f"File: {pdf_files[0].name}")
        print("--- Text Preview ---")
        print(pdf_text[:400] + "...\n")
    else:
        print(f"No PDF files found in {pmd_folder}\n")

    # 2. Process WWIS JSON
    json_file = DATA_DIR / "wmo_city_forecasts" / "lahore_forecast.json"
    print("=== WWIS JSON EXTRACTION ===")
    if json_file.exists():
        parsed_json_text = parse_wwis_json(json_file)
        print(parsed_json_text)
    else:
        print(f"File not found: {json_file}\n")