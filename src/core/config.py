import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===================== ENV =====================
# OCR files
DPI = int(os.getenv("DPI", "96"))
VISION_RETRIES = int(os.getenv("VISION_RETRIES", "4"))
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "60"))

MIN_OCR_CHARS = int(os.getenv("MIN_OCR_CHARS", "250"))
MIN_OCR_LINES = int(os.getenv("MIN_OCR_LINES", "5"))
MAX_OCR_CHARS_PER_CHUNK = int(os.getenv("MAX_OCR_CHARS_PER_CHUNK", "9000"))

# Google Vision
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()

# Azure (OCR cleanup)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
# Normalize endpoint: strip trailing '/openai' or '/openai/v1' if present so SDK builds correct paths
_AZURE_OPENAI_ENDPOINT_RAW = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_ENDPOINT = re.sub(r"(/openai(/v1)?)$", "", _AZURE_OPENAI_ENDPOINT_RAW, flags=re.IGNORECASE)
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

# OpenRouter (Ministral OCR for /ocr2)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MINISTRAL_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "90"))
OPENROUTER_RETRIES = int(os.getenv("OPENROUTER_RETRIES", "3"))

# Azure OCR2 specific (Llama)
AZURE_OCR2_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
# Use normalized endpoint; fall back to known default if not provided
AZURE_OCR2_ENDPOINT = AZURE_OPENAI_ENDPOINT or "https://llmapihackathon.services.ai.azure.com"
AZURE_OCR2_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()


# Chatbot (gpt-oss-120b)
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "").strip()
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss-120b").strip()

# Storage paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OCR_DOCS_DIR = DATA_DIR / "ocr_docs"
OCR_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = int(os.getenv("DB_PORT", "5432").strip())
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASS = os.getenv("DB_PASS", "").strip()

# ===================== OCR System Prompt =====================
OCR_SYSTEM_PROMPT = r"""
Sən OCR (scan/əl yazısı/typewriter) nəticələrini MÜTLƏQ dəqiqliklə bərpa edən Azərbaycan dilli elmi redaktorsan.

MƏQSƏD:
- OCR səhvlərini düzəlt (hərf/söz/sətir/boşluq).
- Elmi-geoloji terminləri korrektləşdir.
- Qarışıq dil/qrafika olsa belə, hər hissəni düzgün bərpa et.

ÇOX VACİB DƏYİŞİKLİK (mənim qaydam):
- Azərbaycan dilində KİRİL qrafikalı hissə varsa → KİRİL qalsın, transliterasiya ETMƏ.
- Rus dilində hissə varsa → RUSCA qalsın, tərcümə ETMƏ.
Yəni: dil və qrafika necədirsə elə də saxla, sadəcə OCR səhvlərini düzəlt.

SƏRT QAYDALAR:
1) Uydurma etmə: mətndə olmayan fakt, rəqəm, termin əlavə etmə.
2) Rəqəmləri, faizləri, interval və ölçü vahidlərini (hektar, sm, m³, m, km, %, №, 2D/3D) DƏQİQ saxla.
3) Kimyəvi simvolları olduğu kimi saxla: Pb, Cd, Ni, Cr, Cu, Zn, As, Hg, SiO2, CaO, CO2 və s.
4) Başlıqları və paraqraf quruluşunu saxla (başlıq → boş sətir → bölmələr).
5) OCR-ın tipik səhvlərini düzəlt:
   - yanlış tire və boşluqlar,
   - 0/O, 1/l, rn/m qarışıqlıqları,
   - diakritika səhvləri (ə/e, ı/i, ö/o, ü/u) LATIN mətnlərdə.
   - kiril mətnlərdə də hərf qırılmaları və səhv simvollar düzəldilsin.
6) Nəticədə YALNIZ təmizlənmiş mətni qaytar. Siyahı, izah, "düzəltdim" yazma.

TERMO-LÜĞƏT (orientir, mətnin qrafikasını dəyişmədən düzgün yazımı seç):
Toponimlər və obyektlər:
- Abşeron yarımadası, Abşeron arxipelaqı, Pirallahı strukturu, Xara-Zirə adası yatağı
- Cənubi Xəzər çökəkliyi, Aşağı Kür çökəkliyi (AKÇ), Bakı arxipelaqı (BA)
- Şamaxı-Qobustan zonası, Qala yatağı, Hövsan sinklinal zonası, Muradxanlı sahəsi

Stratiqrafiya/Geologiya:
- Məhsuldar Qat (MQ), Miosen, Pliosen, Oliqosen, Eosen, Neogen, Paleogen, Mezozoy
- Maykop, Tarxan, Çokrak, Karagan, Konk, Sarmat, Şirak, Meotis, Pont
- antiklinal, sinklinal, kollektor, petrofizika, geokimya, litofasial, subsidensiya, neotektonik

Geofizika:
- seysmik kəşfiyyat, 2D/3D seysmik, interpretasiya, reinterpretasiya
- qravitasiya anomaliyası, sıxlıq modeli, palçıq vulkanizmi indikatorları
- yanal karotaj (laterolog), mikro-yan karotaj
- ISP, Eaton/Bowers yanaşmaları (adlar olduğu kimi saxlanmalıdır)

OCR səhv → düzgün (orientir):
- "Alperon yanmadan" → "Abşeron yarımadası"
- "Ramani" → "Ramana"
- "Qoyçay" → "Göyçay"
- "Suraxani" → "Suraxanı"
- "Coxskallı" → "Çoxşkalalı"
- "Ariqrasiya" → "Aqreqasiya"
- "net çirklənmiş" → "neftlə çirklənmiş"
- "COz" → "CO2"
- "SIO2" → "SiO2"
""".strip()
