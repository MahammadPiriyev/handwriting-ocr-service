import base64
import json
import mimetypes
import time
from typing import List

import requests
import openai
from fastapi import UploadFile

from src.core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    MINISTRAL_MODEL,
    OPENROUTER_TIMEOUT,
    OPENROUTER_RETRIES,
    AZURE_OCR2_API_KEY,
    AZURE_OCR2_ENDPOINT,
    AZURE_OCR2_DEPLOYMENT,
    API_VERSION,
    MIN_OCR_CHARS,
    MIN_OCR_LINES,
    MAX_OCR_CHARS_PER_CHUNK,
)
from src.utils.text_helpers import _score_text, pre_normalize_ocr, chunk_text

# ===================== HELPERS =====================
def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = encode_image_bytes(image_bytes)
    return f"data:{mime_type};base64,{b64}"

def has_latin_english(s: str) -> bool:
    import re
    LAT_EN_RE = re.compile(r"[A-Za-z]")
    return bool(LAT_EN_RE.search(s or ""))

def has_cyrillic(s: str) -> bool:
    import re
    CYR_RE = re.compile(r'[А-Яа-яЁёӘәҒғҺһҸҹӨөҮү]')
    return bool(CYR_RE.search(s or ""))

# ===================== OPENROUTER OCR (MINISTRAL) =====================
MINISTRAL_SYSTEM = (
    "You are an OCR transcription engine. "
    "Extract ALL visible text from the image exactly as written. "
    "Do not translate, do not transliterate, do not summarize. "
    "Keep original scripts (Cyrillic stays Cyrillic, Russian stays Russian, English stays English, Azerbaijani Latin stays Azerbaijani Latin). "
    "Specifically, if you see text in Azerbaijani Cyrillic, it MUST remain in Cyrillic. DO NOT transliterate it to Latin. For example, 'мәлумат' should be 'мәлумат', NOT 'məlumat'."
    "Preserve Azerbaijani Latin diacritics exactly: ə, ğ, ı, ö, ü, ç, ş (DO NOT simplify them). "
    "Preserve punctuation, brackets, slashes, dashes, and scientific notation exactly as written (e.g., Benzo[a]piren, C10–C40, cmol(+)/kg, dS/m, 10^-4). "
    "Preserve capitalization for acronyms (ICP-MS, GC-MS, XRF, ELCR, HQ/HI, m-SEPA, ORC, SVE, VOC, GIS, TPH, PAK/PAH). "
    "Preserve line breaks, headings, bullets/numbering as best as possible. "
    "Ignore the notebook grid/background lines; do not invent underscores or table borders. "
    "Return ONLY the extracted text."
)

MINISTRAL_USER = (
    "Extract all text from this image.\n"
    "Important rules:\n"
    "- No translation or transliteration.\n"
    "- Keep scripts as-is (Cyrillic stays Cyrillic; Russian stays Russian; English stays English; Azerbaijani Latin stays Azerbaijani Latin).\n"
    "- Do NOT convert Cyrillic to Latin. If the text is in Cyrillic (including Azerbaijani Cyrillic), keep it in Cyrillic. For example, 'мәлумат' must be transcribed as 'мәлумат', NOT 'məlumat'.\n"
    "- Keep Azerbaijani diacritics exactly: ə, ğ, ı, ö, ü, ç, ş.\n"
    "- Keep scientific abbreviations and units exactly (ICP-MS, GC-MS/GC-FID, XRF, ELCR, HQ/HI, m-SEPA, ORC, SVE, VOC, GIS, mg/kg, g/kg, cmol(+)/kg, dS/m, m³, °C).\n"
    "- Preserve brackets like Benzo[a]piren and ranges like C10–C40.\n"
    "- Output only the text."
)

def openrouter_ocr_ministral(image_bytes: bytes, mime_type: str = "image/png") -> str:
    if not OPENROUTER_API_KEY or "PASTE_" in OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is invalid.")

    data_url = image_bytes_to_data_url(image_bytes, mime_type)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MINISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": MINISTRAL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MINISTRAL_USER},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        # Handwritten pages can be long; keep a safer ceiling.
        "max_tokens": 3500,
    }

    last_err = None
    for attempt in range(1, OPENROUTER_RETRIES + 1):
        try:
            r = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=OPENROUTER_TIMEOUT,
            )
            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text}")

            js = r.json()
            content = (js.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return (content or "").strip()

        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"OpenRouter OCR error: {last_err}")

# ===================== AZURE CALL: IMAGE + MINISTRAL OCR =====================
SYSTEM_PROMPT = r"""
Sən OCR (scan/əl yazısı/typewriter) nəticələrini MÜTLƏQ dəqiqliklə bərpa edən elmi redaktorsan.

MƏQSƏD:
- ŞƏKİLDƏN mətni maksimum dəqiqliklə çıxart (OCR), o cümlədən:
  • başlıqlar, altbaşlıqlar
  • maddələnmiş siyahılar (1., 2., -, •)
  • mötərizələr, mötərizə içi ifadələr, kvadrat mötərizələr
  • interval/rəqəm/vahidlər (0–5 sm, 5–25 sm, 25–60 sm; 2,980–24,720 m³; 30–45°C; 10^-4 və s.)
  • latın takson adları və elmi qısaltmalar
- Sonra OCR səhvlərini düzəlt (hərf/söz/sətir/boşluq).
- Elmi terminləri korrektləşdir.
- Qarışıq dil/qrafika olsa belə, hər hissəni düzgün bərpa et.

ÇOX VACİB QAYDA (DİL/QRAFİKA DƏYİŞMƏSİN):
- Azərbaycan dilində LATIN qrafikalı hissə varsa → LATIN qalsın; diakritiklər itMƏSİN.
- Azərbaycan dilində KİRİL qrafikalı hissə varsa → KİRİL qalsın, AZƏRBAYCAN LATIN QRAFİKASINA ÇEVİRMƏ. (Məsələn: "Петрофизики параметрлер" → Kiril qalsın, "Petrofiziki parametrlər" yazma.)
- Rus dilində hissə varsa → RUSCA (Kiril) qalsın, tərcümə ETMƏ və AZƏRBAYCAN LATIN QRAFİKASINA ÇEVİRMƏ.
- İngilis dilində hissə varsa → ENGLISH qalsın, tərcümə ETMƏ.
Yəni: dil və qrafika necədirsə elə də saxla, sadəcə OCR səhvlərini düzəlt.

AZƏRBAYCAN LATIN DİAKRİTİKLƏRİ (MÜTLƏQ QORUNSUN):
- ə, ğ, ı, ö, ü, ç, ş
Tipik səhvlər: ə→e/a, ğ→g/q, ı→i/l, ö→o, ü→u, ç→c, ş→s.
Bu hərfləri "sadələşdirmə" — şəkildən necə oxunursa elə yaz.

SƏRT QAYDALAR:
1) Uydurma etmə: mətndə olmayan fakt, rəqəm, termin əlavə etmə.
2) Rəqəmləri, faizləri, interval və ölçü vahidlərini (hektar/ha, sm, m³, °C, dS/m, cmol(+)/kg, g/kg, mg/kg, %, №) DƏQİQ saxla.
3) Kimyəvi simvolları olduğu kimi saxla: Pb, Cd, Ni, Cr, Cu, Zn, As, Hg, SiO2, CO2, H2S və s.
4) Başlıqları ve paraqraf quruluşunu saxla.
5) OCR-ın tipik səhvlərini düzəlt:
   - yanlış tire ve boşluqlar (C10–C40, 2,980–24,720 m³ kimi),
   - 0/O, 1/l, rn/m qarışıqlıqları,
   - elmi qısaltmalarda hərf sürüşməsi (lCP-MS → ICP-MS, 0RC → ORC, SⅤE → SVE və s.),
   - "m-SEPA" kimi defisləri, "HQ/HI" kimi slash-ları qoruma,
   - kvadrat mötərizə ve indekslər: Benzo[a]piren → mötərizə/simvol itMƏSİN,
   - latın takson adlarında böyük/kiçik hərf düzgün qalsın (Aspergillus, Trichoderma, Penicillium və s.).
6) Nəticədə YALNIZ təmizlənmiş mətni qaytar. Siyahı, izah, "düzəltdim" yazma.
7) Mətnin oxunma sırası: əvvəl səhifənin yuxarı hissəsi, sonra əsas mətn, sonra alt hissə.
8) Dəftər xətləri/şəbəkə fonu ve ləkələri mətndən sayma (underscore, "| | |", saxta cədvəl xətti çıxartma).

========================================
TERMO-LÜĞƏT / ORİYENTİR (ŞƏKİLDƏ OLANI DÜZGÜN YAZMAQ ÜÇÜN)
========================================
DIQQƏT: Aşağıdakı termo-lüğət yalnız LATIN qrafikada olan mətn üçündür. KİRİL qrafikada olan mətn üçün transliterasiya etmə - orijinal Kiril qrafikasını saxla.

Toponimlər ve obyektlər:
- Abşeron yarımadası
- Ramana, Suraxanı
- SOCAR
- Bakı Yod Zavodu

Açar mövzular ve terminlər:
- torpaq çirklənməsi, sənaye çirklənməsi
- ağır metallar, karbohidrogenlər
- neft məhsulları, karbohidrogen çirklənməsi
- polisiklik aromatik karbohidrogenlər (PAK) / PAH
- geotermal sular, geotermal enerji
- bioremediasiya, fitoremediasiya, elektrokinetik remediya
- biostimulyasiya, biosurfaktant, nutrient optimallaşdırılması
- bioakkumulyasiya, biodeqradasiya
- immobilizasiya, remobilizasiya, karbonat tamponlanması, gil mineralları
- "qaynar nöqtələr" (hotspots), areal (yayılmış) çirklənmə, lokal (nöqtəvi) çirklənmə
- külək–səth axını, topoqrafik çökəkliklər

Elmi metodlar, qısaltmalar ve analitika (BÖYÜK/kiçik hərf DƏQİQ):
- QA/QC
- pH (H2O), pH (KCl)
- EC (elektrik keçiriciliyi), dS/m
- LOI (üzvi maddə), tekstura (hidrometr)
- KMT (kation-mübadilə tutumu) / CEC (əgər belə yazılıbsa)
- TPH (Total Petroleum Hydrocarbons)
- PAK (PAH)
- VOC
- GC-MS, GC-FID, GC–MS/GC–FID (şəkildə necədirsə)
- C10–C40 fraksiyalaşdırma (dash ve aralıq itMƏSİN)
- XRF
- ICP-MS
- DTPA (DTPA-ekstrakt)
- Kriging, IDW interpolasiya
- Spearman korrelyasiya
- PCA, PC1, PC2
- HQ, HI (HQ/HI), ELCR
- m-SEPA
- GIS
- ORC (Organic Rankine Cycle)
- SVE (Soil Vapor Extraction) / torpaq buxar (vakuum) ekstraksiyası (şəkildə necə yazılıbsa)

Vahidlər ve formatlar (dəqiq saxla):
- ha/hektar (məs: 222,000 ha; 33,300 ha; 7,500 ha)
- sm (0–5 sm; 5–25 sm; 25–60 sm)
- m³ (məs: 2,980–24,720 m³)
- °C (məs: 30–45°C)
- g/kg, mg/kg
- cmol(+)/kg
- 10^-x (məs: 10^-4, 10^-6 kimi — üstlük formatını itirmə)

Kimyəvi/çirkləndirici adları (şəkildə olanı seç):
- fenantren (phenanthrene), antrasen (anthracene)
- Benzo[a]piren (mötərizə ve "a" saxlanılsın)
- H2S, SiO2, CO2

Taksonomiya / mikobiota (latınca yazılışı qoru):
- Ascomycota, Basidiomycota, Mucoromycota
- Penicillium, Aspergillus, Fusarium, Trichoderma
- Trichoderma harzianum, Aspergillus niger (əgər mətn belədirsə)
- Coprinellus, Trametes
- Mucor, Rhizopus
- metallotionein / metallothionein (şəkildə necədirsə)

Bitki adları (əgər mətn belədirsə, düzgün saxla; yoxdursa uydurma):
- Atriplex, Phragmites, Halostachys (ve s.)

========================================
OCR səhv → düzgün (orientir; yenə də yalnız şəkildən oxuna biləni yaz)
========================================
Diakritiklər:
- "Abseron/Absheron" → "Abşeron"
<<<<<<< HEAD
- "yanmasında" → "yarımadasında"
- "yanmalarında" → "yarımadasında"
- "yaşmasında" → "yarımadasında"
=======
>>>>>>> d8cc972b6bdc1329050d6af9e91d95dab9a76734
- "Suraxani" → "Suraxanı"
- "Ramani/Ramani" → "Ramana"
- "bioremediasiya/bioremediasiya" → "bioremediasiya" (diakritiklər qorunsun)

Qısaltmalar:
- "lCP-MS / ICP–MS" → "ICP-MS"
- "XRE / XRF" → "XRF"
- "GCMS / GC-MS" → "GC-MS"
- "GCFID / GC-FID" → "GC-FID"
- "ELRC / ELC R / ELCR" → "ELCR"
- "H0 / HQ" → "HQ"
- "Hl / HI" → "HI"
- "mSEPA / m – SEPA" → "m-SEPA"
- "0RC" → "ORC"
- "SVF" → "SVE"
- "GISS / GİS" → "GIS"
- "SOC AF / SOCAF" → "SOCAR"

Vahidlər:
- "cmol(+)/kg" içində "(+)" itMƏSİN
- "dS/m" içində "/" itMƏSİN
- "m3" → "m³" (əgər üstlüq görünürsə)
- "30-45C" → "30–45°C" (əgər °C yazılıbsa)

DİQQƏT:
- Şəkildə olmayan "düzəliş" etmə. Əmin olmadığın sözlərdə fantaziya etmə — ən yaxın oxunuşu saxla, amma yeni termin artırma.
- KİRİL qrafikasında olan mətn üçün transliterasiya etmə - orijinal Kiril qrafikasını saxla.

Nəticədə YALNIZ təmizlənmiş mətni qaytar.
"""

def azure_fix_with_image_and_ministral(
    ocr_text: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    pass_hint: str = "",
) -> str:
    if not AZURE_OCR2_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY (AZURE_OCR2_API_KEY) is not set.")
    if not AZURE_OCR2_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT (AZURE_OCR2_ENDPOINT) is not set.")

    client = openai.AzureOpenAI(
        api_key=AZURE_OCR2_API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_OCR2_ENDPOINT,
    )

    data_url = image_bytes_to_data_url(image_bytes, mime_type)

    user_text = f"""
Sənə eyni mənbədən iki siqnal verilir:
1) OpenRouter (mistralai/ministral-14b-2512) OCR mətni (xam OCR) — bəzən diakritikləri ve qısaltmaları korlaya bilər
2) Orijinal şəkil — buradan MÜTLƏQ bütün mətni oxu ve dəqiqləşdir

Tapşırıq:
- Şəkildən də OCR et ve Ministral OCR ilə MÜQAYİSƏ ET.
- Qısaltmalar, latınca adlar, kimyəvi simvollar, vahidlər, mötərizələr, üstlüklər dəqiq saxlanmalıdır.
- Dil/qrafika qaydasını pozma: LATIN latında qalsın (ə/ğ/ı/ö/ü/ç/ş qorunsun), KİRİL kirildə qalsın, RUS rusca qalsın, EN ingiliscə qalsın.
- Xüsusi qeyd: Azərbaycan dilində yazılmış Kiril mətni AZƏRBAYCAN LATIN QRAFİKASINA ÇEVİRMƏ. Orijinal Kiril qrafikasında saxla.
- Xüsusi qeyd 2: Rus dilində yazılmış Kiril mətni də orijinal Kiril qrafikasında saxla.
- Transliterasiya YOXDUR: Kiril mətni heç bir halda Latın qrafikasına çevirmə.
- Uydurma etmə; rəqəmləri/ölçüləri/kimyəvi simvolları DƏQİQ saxla.
- Sonda YALNIZ yekun təmizlənmiş mətni qaytar.

{pass_hint}

--- MINISTRAL OCR (XAM) ---
{ocr_text}
--- SON ---
""".strip()

    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=AZURE_OCR2_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=5000,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Azure OCR2 call failed: {last_err}")

def run_ocr2_pipeline(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Main entry point for OCR2 logic.
    1. OpenRouter (Ministral)
    2. Normalize & Chunk
    3. Azure (Image + Text) repair pass per chunk
    """
    # 1) OpenRouter
    ministral_raw = openrouter_ocr_ministral(image_bytes, mime_type)
    chars, lines = _score_text(ministral_raw)

    if not ministral_raw.strip():
        return ""

    # 2) Normalize & chunk
    ocr_ready = pre_normalize_ocr(ministral_raw)
    parts = chunk_text(ocr_ready, MAX_OCR_CHARS_PER_CHUNK)

    # 3) Azure repair per part
    cleaned_parts = []
    for i, part in enumerate(parts, 1):
        cleaned_parts.append(
            azure_fix_with_image_and_ministral(
                ocr_text=part,
                image_bytes=image_bytes,
                mime_type=mime_type,
                pass_hint=f"(Hissə {i}/{len(parts)})",
            )
        )

    cleaned = "\n\n".join([p for p in cleaned_parts if p]).strip()

    # Final pass: if raw had Cyrillic but cleaned lost it, force another image-guided pass
    # with even stronger instructions
    if has_cyrillic(ocr_ready) and (not has_cyrillic(cleaned)):
        # Create a more explicit instruction for Cyrillic preservation
        cyrillic_hint = """(FINAL PASS - CRITICAL INSTRUCTION)
        Diqqət: Bu səhifədə Kiril qrafikasında yazılmış mətn var. 
        Bu Kiril mətni AZƏRBAYCAN LATIN QRAFİKASINA ÇEVİRMƏ.
        Orijinal Kiril qrafikasını saxla.
        Qayda: Hər hansı Kiril mətn (Azərbaycan və ya Rus dili) → Orijinal Kiril qrafikası saxlanılmalıdır.
        Transliterasiya YOXDUR: Kiril mətni heç bir halda Latın qrafikasına çevirmə.
        Nümunə: "Петрофизики параметрлер" → "Петрофизики параметрлер" (Kiril qalsın)
        Nümunə: "Нефтяная скважина" → "Нефтяная скважина" (Kiril qalsın)"""
        
        cleaned = azure_fix_with_image_and_ministral(
            ocr_text=cleaned,
            image_bytes=image_bytes,
            mime_type=mime_type,
            pass_hint=cyrillic_hint,
        )

    return cleaned