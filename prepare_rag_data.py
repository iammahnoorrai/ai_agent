import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw" / "weather_forecasts"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

def extract_pdf_text(pdf_path: Path) -> str:
    """Extracts clean text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def parse_wwis_json(json_path: Path) -> str:
    """Parses WWIS JSON into structured text for vector embedding."""
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

def build_vector_store():
    documents = []

    # 1. Load PMD PDF
    pmd_folder = DATA_DIR / "pmd_daily_forecasts"
    pdf_files = list(pmd_folder.glob("*.pdf"))
    if pdf_files:
        pdf_text = extract_pdf_text(pdf_files[0])
        documents.append(f"Source: PMD Daily Forecast Bulletin\n{pdf_text}")

    # 2. Load WWIS JSON
    json_file = DATA_DIR / "wmo_city_forecasts" / "lahore_forecast.json"
    if json_file.exists():
        json_text = parse_wwis_json(json_file)
        documents.append(json_text)

    print(f"Loaded {len(documents)} document source(s).")

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 4. Generate Embeddings & Save Vector Store
    print("Generating embeddings (using sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(VECTOR_STORE_DIR))
    print(f"✅ Vector store successfully saved to: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    build_vector_store()