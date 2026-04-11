import io
import base64
import json
import re
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import pytesseract
from pathlib import Path
import shutil
import unicodedata

# ─── FIXED: fall back to the hardcoded path if not found in system PATH ───
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseract_path = shutil.which("tesseract") or pytesseract.pytesseract.tesseract_cmd
print(tesseract_path)

if not tesseract_path or not Path(tesseract_path).exists():
    raise RuntimeError(
        "Tesseract not found. Please install it:\n"
        "Mac: brew install tesseract\n"
        "Ubuntu: sudo apt install tesseract-ocr\n"
        "Windows: https://github.com/tesseract-ocr/tesseract"
    )

pytesseract.pytesseract.tesseract_cmd = tesseract_path

def normalize_text(text: str) -> str:
    """
    General-purpose text normalizer for all extracted content.
    Safe to run on OCR output, PDF text, PPTX body text.
    """
    # Normalize unicode (curly quotes → straight, ligatures → ascii)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove control chars and zero-width chars
    text = re.sub(
        r'[\x00\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\ufeff]', '', text
    )

    # Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    lines       = text.split('\n')
    clean_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            clean_lines.append('')
            continue
        # Skip horizontal rules
        if re.fullmatch(r'[-=_*]{3,}', s):
            continue
        # Skip bare page numbers
        if re.fullmatch(r'\d{1,3}', s) and len(lines) > 20:
            continue
        # Skip lines that are a single character repeated
        if len(set(s)) == 1 and len(s) > 2:
            continue
        # Skip lines where <30% of characters are alphabetic (OCR garbage)
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.25 and len(s) > 12 and not re.search(r'[a-zA-Z]', s):
            continue
        clean_lines.append(line)

    return '\n'.join(clean_lines).strip()

def clean_ocr_text(text: str) -> str:
    """Enhanced OCR post-processing — calls normalize_text then applies OCR-specific rules."""
    text  = normalize_text(text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        lowered = line.strip().lower()
        # Strip LLM preamble that leaks through reconstruct_code_llm
        if any(p in lowered for p in (
            "here is", "i fixed", "the corrected", "below is",
            "certainly", "sure,", "of course"
        )):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def ocr_image(image_bytes):
    try:
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.point(lambda x: 0 if x < 140 else 255, '1')
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        text = pytesseract.image_to_string(img, config='--psm 4 -c preserve_interword_spaces=1')

        text = clean_ocr_text(text)
        return text.strip()
    except Exception as e:
        print("[OCR ERROR]:", e)
        return ""
    
# ─────────────────────────────────────────────────────────────────────────────
# Image Type Classification
# ─────────────────────────────────────────────────────────────────────────────
def detect_code(text):

    lines = text.split("\n")
    score = 0

    for line in lines:
        line = line.strip()

        # STRONG indicators only
        if (
            line.startswith("class ") or
            line.startswith("public ") or
            line.startswith("private ") or
            line.startswith("def ") or
            line.startswith("#include") or
            "::" in line or
            "->" in line or
            ("(" in line and ")" in line and "{" in line) or
            ("=" in line and ";" in line)
        ):
            score += 2   # strong signal

        # WEAK indicators
        elif (
            "{" in line or "}" in line or ";" in line
        ):
            score += 1

    return score >= 4


def classify_image_type(image_bytes: bytes) -> str:
    """
    Classify image into: 'table', 'chart', 'code', 'diagram', 'text', 'other'
    Uses lightweight CV heuristics — no deep model needed.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return "other"

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w   = gray.shape

        # ── Detect monospace/code regions ──
        # Code blocks tend to have uniform left-margin and regular character spacing
        # Use text density in the left 20% of image as a proxy
        left_strip   = gray[:, :w // 5]
        left_density = np.sum(left_strip < 128) / (h * w // 5)
        if left_density > 0.05:
            pil_img      = Image.fromarray(gray)
            ocr_sample   = pytesseract.image_to_string(pil_img, config='--psm 6')
            code_signals = ['{', '}', ';', '::', '->', 'def ', 'class ', '#include', 'return', 'void']
            code_count   = sum(1 for s in code_signals if s in ocr_sample)
            if code_count >= 3:
                return "code"
            
        else:
            pil_img      = Image.fromarray(gray)
            ocr_sample   = pytesseract.image_to_string(pil_img, config='--psm 6')
            print(ocr_sample)
            detect = detect_code(ocr_sample)
            print(detect)
            if detect:
                return "code"

        # # ── Detect bars/blocks of colour (chart indicator) ──
        # hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # saturation  = hsv[:, :, 1]
        # high_sat    = np.sum(saturation > 80)
        # total_px    = h * w
        # sat_ratio   = high_sat / total_px

        # if sat_ratio > 0.15:
        #     # Check for bar-like rectangular blobs
        #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #     _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     rect_count = sum(
        #         1 for c in contours
        #         if cv2.contourArea(c) > 500 and _is_rectangular(c)
        #     )
        #     if rect_count >= 3:
        #         return "chart"
        #     edges = cv2.Canny(gray, 50, 150)
        #     edge_ratio = np.sum(edges > 0) / (h * w)

        #     if edge_ratio > 0.02:
        #         return "chart"

        # # ── Detect horizontal/vertical lines (table indicator) ──
        # horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        # vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
        # horiz_lines  = cv2.morphologyEx(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], cv2.MORPH_OPEN, horiz_kernel)
        # vert_lines   = cv2.morphologyEx(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], cv2.MORPH_OPEN, vert_kernel)

        # n_horiz = cv2.countNonZero(horiz_lines)
        # n_vert  = cv2.countNonZero(vert_lines)

        # if n_horiz > (w * 0.5) and n_vert > (h * 0.5):
        #     return "table"

        # ── Default: treat as text/diagram ──
        return "text"

    except Exception as e:
        print(f"[classify_image_type] Error: {e}")
        return "other"


def _is_rectangular(contour) -> bool:
    """Check if a contour is roughly rectangular (for chart bar detection)."""
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    return len(approx) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Table Extraction from Images
# ─────────────────────────────────────────────────────────────────────────────

# def extract_table_from_image(image_bytes: bytes) -> str:
#     """
#     Extracts tabular data from an image using line detection + cell OCR.
#     Returns a markdown-style table string.
#     """
#     try:
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         h, w   = gray.shape

#         # Threshold
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         # Detect horizontal lines
#         horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                                  cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1)))
#         # Detect vertical lines
#         vert  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                                  cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5)))

#         # Find grid intersections
#         grid      = cv2.add(horiz, vert)
#         contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#         # Sort bounding boxes top→bottom, left→right
#         cells = []
#         for c in contours:
#             x, y, cw, ch = cv2.boundingRect(c)
#             if cw > 20 and ch > 10:
#                 cells.append((y, x, cw, ch))
#         cells.sort(key=lambda c: (c[0] // 20, c[1]))  # row-major

#         if not cells:
#             # Fallback: OCR the whole image as text
#             pil = Image.fromarray(gray)
#             return pytesseract.image_to_string(pil, config='--psm 6').strip()

#         # OCR each cell
#         pil_img   = Image.fromarray(img)
#         row_texts = []
#         prev_row  = -1
#         current   = []

#         for (y, x, cw, ch) in cells:
#             row_id = y // 25
#             if prev_row != -1 and row_id != prev_row:
#                 row_texts.append(current)
#                 current = []
#             cell_crop = pil_img.crop((x, y, x + cw, y + ch))
#             cell_text = pytesseract.image_to_string(
#                 cell_crop, config='--psm 7'
#             ).strip().replace('\n', ' ')
#             current.append(cell_text)
#             prev_row = row_id

#         if current:
#             row_texts.append(current)

#         if not row_texts:
#             return ""

#         # Format as markdown table
#         max_cols = max(len(r) for r in row_texts)
#         lines    = []
#         for i, row in enumerate(row_texts):
#             padded = row + [""] * (max_cols - len(row))
#             lines.append("| " + " | ".join(padded) + " |")
#             if i == 0:
#                 lines.append("|" + " --- |" * max_cols)

#         return "[TABLE]\n" + "\n".join(lines)

#     except Exception as e:
#         print(f"[extract_table_from_image] Error: {e}")
#         return ""

def extract_table_from_image(image_bytes: bytes) -> str:
    try:
        # --- Step 1: OCR ---
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Improve contrast (helps OCR a lot)
        pil = ImageEnhance.Contrast(pil).enhance(2.0)

        raw_text = pytesseract.image_to_string(pil, config='--psm 6')
        raw_text = clean_ocr_text(raw_text)

        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        if len(lines) < 2:
            return "[TABLE]\n" + raw_text

        # --- Step 2: Smart row splitting ---
        def split_row(line):
            # Try splitting by multiple spaces first
            parts = re.split(r'\s{2,}', line)

            # Fallback: split by single space
            if len(parts) <= 1:
                parts = line.split()

            return [p.strip() for p in parts if p.strip()]

        rows = [split_row(line) for line in lines]

        # --- Step 3: Find dominant column count ---
        from collections import Counter

        counts = [len(r) for r in rows if len(r) > 0]
        if not counts:
            return "[TABLE]\n" + raw_text

        target_cols = Counter(counts).most_common(1)[0][0]

        # --- Step 4: Normalize rows ---
        normalized = []
        for r in rows:
            if len(r) == target_cols:
                normalized.append(r)
            elif len(r) > target_cols:
                normalized.append(r[:target_cols])
            else:
                normalized.append(r + [""] * (target_cols - len(r)))

        # --- Step 5: Detect header ---
        def is_header(row):
            alpha_count = sum(
                any(c.isalpha() for c in cell) for cell in row
            )
            return (alpha_count / max(len(row), 1)) > 0.5

        if not is_header(normalized[0]):
            normalized.insert(0, [f"Column {i+1}" for i in range(target_cols)])

        # --- Step 6: Build markdown table ---
        lines_out = []

        for i, row in enumerate(normalized):
            lines_out.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines_out.append("|" + " --- |" * target_cols)

        return "[TABLE]\n" + "\n".join(lines_out)

    except Exception as e:
        print(f"[extract_table_from_image] Error: {e}")
        return ""
# def extract_table_from_image(image_bytes: bytes) -> str:
#     try:
#         pil = Image.open(io.BytesIO(image_bytes)).convert("L")

#         # Improve contrast
#         pil = ImageEnhance.Contrast(pil).enhance(2.0)

#         # OCR entire image
#         text = pytesseract.image_to_string(pil, config='--psm 6')

#         text = clean_ocr_text(text)

#         lines = [l.strip() for l in text.split("\n") if l.strip()]

#         # Try to align columns
#         rows = []
#         for line in lines:
#             parts = re.split(r'\s{2,}', line)  # split by multiple spaces
#             rows.append(parts)

#         # Normalize column count
#         max_cols = max(len(r) for r in rows) if rows else 0

#         formatted = []
#         for i, row in enumerate(rows):
#             row = row + [""] * (max_cols - len(row))
#             formatted.append("| " + " | ".join(row) + " |")
#             if i == 0:
#                 formatted.append("|" + " --- |" * max_cols)

#         return "[TABLE]\n" + "\n".join(formatted)

#     except Exception as e:
#         print(f"[extract_table_from_image] Error: {e}")
#         return ""


# ─────────────────────────────────────────────────────────────────────────────
# Chart / Graph Data Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_chart_data(image_bytes: bytes) -> str:
    """
    Extracts text labels, axis info, and legend from a chart image.
    Uses Tesseract in layout-aware mode + colour region analysis.
    Returns a structured description the LLM can reason over.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w   = gray.shape

        pil = Image.fromarray(img)

        # ── Step 1: Extract all text with bounding boxes ──
        data = pytesseract.image_to_data(
            Image.fromarray(gray),
            config='--psm 11',
            output_type=pytesseract.Output.DICT
        )

        texts = []
        for i, txt in enumerate(data['text']):
            if txt.strip() and int(data['conf'][i]) > 40:
                x, y = data['left'][i], data['top'][i]
                texts.append((y, x, txt.strip()))

        # ── Step 2: Classify text by position ──
        # Title = top 15% of image; X-axis = bottom 15%; Y-axis = left 15%; Legend = right 20%
        title_texts  = [t for t in texts if t[0] < h * 0.15]
        bottom_texts = [t for t in texts if t[0] > h * 0.80]
        left_texts   = [t for t in texts if t[1] < w * 0.15]
        legend_texts = [t for t in texts if t[1] > w * 0.78]
        body_texts   = [t for t in texts if
                        h * 0.15 <= t[0] <= h * 0.80 and
                        w * 0.15 <= t[1] <= w * 0.78]

        def join_texts(lst):
            return " ".join(t[2] for t in sorted(lst))

        title  = join_texts(title_texts)
        x_axis = join_texts(bottom_texts)
        y_axis = join_texts(left_texts)
        legend = join_texts(legend_texts)
        body   = join_texts(body_texts)

        # ── Step 3: Count dominant colour regions (approximate bar/pie count) ──
        hsv          = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat          = hsv[:, :, 1]
        val          = hsv[:, :, 2]
        coloured_px  = np.sum((sat > 60) & (val > 60))
        region_ratio = coloured_px / (h * w)

        chart_desc = []
        if title:
            chart_desc.append(f"Chart title: {title}")
        if x_axis:
            chart_desc.append(f"X-axis labels: {x_axis}")
        if y_axis:
            chart_desc.append(f"Y-axis labels: {y_axis}")
        if legend:
            chart_desc.append(f"Legend: {legend}")
        if body:
            chart_desc.append(f"Data labels visible: {body}")
        chart_desc.append(f"Approximate coloured area: {region_ratio:.0%} of chart")

        return "[CHART]\n" + "\n".join(chart_desc)

    except Exception as e:
        print(f"[extract_chart_data] Error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Code Block Extraction from Images
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_from_image(image_bytes: bytes) -> str:
    """
    Extracts code from screenshot images.
    Applies preprocessing tuned for monospace fonts and dark backgrounds.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil)

        # Detect dark background (common in code screenshots)
        mean_val = np_img.mean()
        if mean_val < 128:
            # Invert for Tesseract (works better on light background)
            np_img = 255 - np_img
            pil    = Image.fromarray(np_img)

        # Convert to grayscale + sharpen
        gray    = pil.convert("L")
        sharp   = gray.filter(ImageFilter.SHARPEN)
        enhanced = ImageEnhance.Contrast(sharp).enhance(2.0)

        # Binarise
        thresh = enhanced.point(lambda x: 0 if x < 140 else 255, '1')

        # Tesseract in single-block mode (best for code)
        raw = pytesseract.image_to_string(thresh, config='--psm 6 -c preserve_interword_spaces=1')

        # Post-process: restore common OCR mistakes in code
        substitutions = [
            (r'\b0\b(?=[a-zA-Z])', 'O'),   # 0 before letter → O
            (r'(?<=[a-zA-Z])0\b', 'O'),    # O before boundary
            (r'\bl\b', '1'),               # lone l → 1
            (r'(?<=\d)l(?=\d)', '1'),      # l between digits
        ]
        for pattern, repl in substitutions:
            raw = re.sub(pattern, repl, raw)

        # Clean and wrap
        cleaned = clean_ocr_text(raw)
        if not cleaned:
            return ""

        return f"[CODE]\n```\n{cleaned}\n```"

    except Exception as e:
        print(f"[extract_code_from_image] Error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Master Image Extractor (replaces the old ocr_image call)
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_content(image_bytes: bytes) -> tuple[str, str]:
    """
    Classifies image type and runs the appropriate extractor.

    Returns:
        (extracted_text, chunk_type)
        chunk_type: "table" | "chart" | "code" | "image" | "text"
    """
    img_type = classify_image_type(image_bytes)
    print(f"  [Image] Classified as: {img_type}")

    if img_type == "table":
        text = extract_table_from_image(image_bytes)
        return text, "table"

    elif img_type == "chart":
        text = extract_chart_data(image_bytes)
        return text, "chart"

    elif img_type == "code":
        text = extract_code_from_image(image_bytes)
        return text, "code"

    else:
        # General OCR for diagrams, text images
        text = ocr_image(image_bytes)
        return text, "image"