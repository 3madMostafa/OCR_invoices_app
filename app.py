# -*- coding: utf-8 -*-
"""
PDF Processing Streamlit App - Advanced PO Extraction v48
A web interface for processing PDF files and extracting structured data
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import os, re, time, PyPDF2, traceback, unicodedata
import tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from io import BytesIO

# Arabic fix (optional but recommended)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    AR_FIX_OK = True
except Exception:
    arabic_reshaper = None
    get_display = None
    AR_FIX_OK = False

# OCR optional
try:
    import pytesseract
    from pdf2image import convert_from_path
    TESS_OK = True
except Exception:
    pytesseract = None
    convert_from_path = None
    TESS_OK = False

# Set page config
st.set_page_config(
    page_title="PDF Data Extractor v48",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= v48 ADVANCED PO EXTRACTION =================

# Normalization constants
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
BIDI_MARKS = "".join(chr(c) for c in [0x200E,0x200F,0x202A,0x202B,0x202C,0x202D,0x202E])

def strip_combining(s: str) -> str:
    """Remove combining marks (diacritics)"""
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_advanced(s: str) -> str:
    """Advanced normalization for PO extraction"""
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_combining(s)
    s = s.replace("\u00A0", " ")
    s = s.translate(ARABIC_DIGITS)
    for ch in BIDI_MARKS:
        s = s.replace(ch, "")
    s = (s.replace("：", ":").replace("–", "-").replace("—", "-").replace("ـ", " ")
           .replace("(", " ").replace(")", " ").replace("["," ").replace("]"," "))
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

# v48 Regex patterns for PO extraction
RX_PO_PREFIX = re.compile(r"\bpo(?:\s*no\.?| reference)?\s*[:#/\-]?\s*(\d{4,6})(?:[^\d\s]*)?(?:\s*[-/]\s*(\d{4,6}))?\b", re.IGNORECASE)
RX_TWO_NUMS = re.compile(r"\b(\d{4,6})\s*[-/]\s*(\d{4,6})\b")
RX_SINGLE_NUM = re.compile(r"\b(\d{4,6})\b")

RX_CODE_ANCHOR_LINE = re.compile(r"(?:\bcode\b|كود)\s*[/:\-]?\s*(?:no\.?|number)?\s*(\d{4,6})", re.IGNORECASE)
RX_CODE_SLASH_NUM = re.compile(r"كود\s*(\d{4,6})\s*/", re.IGNORECASE)
RX_CODE_NO_SPACE = re.compile(r"(\d{4,6})\s*/?\s*كود", re.IGNORECASE)
RX_BOOKING_ORDER = re.compile(r"(?:\border\s*(?:no\.?|number)?\b|اوردر\s*رقم|حجز\s*رقم)\s*[:#/\-]?\s*(\d{4,6})", re.IGNORECASE)
RX_ARABIC_D_PREFIX = re.compile(r"(?:\bد\.?\s*)(\d{4,6})")
RX_ARABIC_D_SUFFIX = re.compile(r"(\d{4,6})\s*د\.")

# v48 New patterns
RX_PO_REF_WITH_ARABIC = re.compile(r"po\s*reference\s*[:#]?\s*(\d{4,6})(?:[^\d\s]*)?", re.IGNORECASE)
RX_PO_SLASH = re.compile(r"\bPO\s*/\s*(\d{4,6})\b", re.IGNORECASE)
RX_PO_FORWARD_SLASH = re.compile(r"\bpo\s*/\s*(\d{4,6})\b", re.IGNORECASE)
RX_PO_NUM_COLON = re.compile(r"\bPO\s*NUM\s*:\s*(\d{4,6})\b", re.IGNORECASE)
RX_PARENTHESES_NUM = re.compile(r"\((?:[^)]*?)(\d{4,6})(?:[^)]*?)\)")
RX_PARENTHESES_NUM_END = re.compile(r"\(.*?(\d{4,6})\s*\)$")
RX_TAXPAYER_NAME_NUM = re.compile(r"taxpayer\s*name\s*[:#]?\s*(\d{4,6})", re.IGNORECASE)
RX_NUM_DATE = re.compile(r"\b(\d{4,6})\s+\d{1,2}/\d{1,2}/\d{4}\b")
RX_NUM_CONCAT_DATE = re.compile(r"\b(\d{6})\d{1,2}/\d{1,2}/\d{4}\b")
RX_STANDALONE_6DIGIT = re.compile(r"\b(\d{6})\s*$")
RX_ARABIC_MEALS_NUM = re.compile(r"وجبات\s*غذائي[هة]\s+(?:د\.\s*)?(\d{4,6})", re.IGNORECASE)

ANCHOR_WORDS = re.compile(r"(?:\bcode\b|كود|كود/|كود:)", re.IGNORECASE)
ANCHOR_WORDS_EXTENDED = re.compile(r"(?:\bcode\b|كود)", re.IGNORECASE)
ORDER_WORDS = re.compile(r"(?:\border\s*(?:no\.?|number)?\b|(?:[أا]وردر)\s*رقم)", re.IGNORECASE)

def find_num_after_anchor(text: str, anchor_rx: re.Pattern, window: int = 400):
    for m in anchor_rx.finditer(text):
        chunk = text[m.end(): m.end()+window]
        mnum = re.search(r"(\d{4,6})", chunk, re.IGNORECASE|re.DOTALL)
        if mnum:
            return mnum.group(1)
    return None

def find_num_before_anchor(text: str, anchor_rx: re.Pattern, window: int = 120):
    for m in anchor_rx.finditer(text):
        start = max(0, m.start()-window)
        chunk = text[start:m.start()]
        mnum = re.search(r"(\d{4,6})\s*(?:د\.?)?$", chunk, re.IGNORECASE|re.DOTALL)
        if mnum:
            return mnum.group(1)
    return None

def find_num_after_anchor_extended(text: str, window: int = 500):
    """Find numbers after كود with more flexible patterns"""
    for m in ANCHOR_WORDS_EXTENDED.finditer(text):
        chunk = text[m.end(): m.end()+window]
        mnum = re.search(r"[:\-/\s]*(\d{4,6})\s*/?", chunk, re.IGNORECASE|re.DOTALL)
        if mnum:
            return mnum.group(1)
    return None

# v48 Label and block utilities
LABEL_TOKENS = (
    "po reference", "proforma invoice number", "so reference",
    "recipients (to)", "recipients to", "taxpayer name",
    "code name", "description", "total sales", "total amount",
    "taxpayer activity code", "submission date", "issuance date",
    "signed by", "internal id", "id:", "unit price", "quantity/", "unit type"
)

def is_label_line(low: str) -> bool:
    return any(tok in low for tok in LABEL_TOKENS)

def collect_block(lines, start_idx, max_lines=180) -> str:
    block = []
    i = start_idx + 1
    while i < len(lines) and len(block) < max_lines:
        low = lines[i].lower()
        if is_label_line(low):
            break
        block.append(lines[i])
        i += 1
    return "\n".join(block)

def slice_between(lines, start_token, stop_tokens=("code name","description","total sales","total amount","taxpayer activity code","po reference","so reference","proforma invoice number","recipients (to)","recipients to","taxpayer name","unit price","quantity/","unit type")):
    out = []
    started = False
    for raw in lines:
        line = norm_advanced(raw)
        low = line.lower()
        if not started and start_token in low:
            started = True
            continue
        if started:
            if any(tok in low for tok in stop_tokens):
                break
            out.append(line)
    return "\n".join(out)

def collect_forbidden_numbers(text: str) -> set:
    """Collect numbers that should not be considered as PO numbers"""
    forbid = set()
    for m in re.finditer(r"taxpayer\s+activity\s+code\s*:\s*(\d{4,6})", text, re.IGNORECASE):
        forbid.add(m.group(1))
    return forbid

def extract_from_po_reference(lines, debug=False):
    """Extract PO from PO Reference section - v48"""
    out = []
    for i, raw in enumerate(lines):
        line = norm_advanced(raw)
        low = line.lower()
        if "po reference" in low:
            # Extract the value directly from this line first
            m = RX_PO_REF_WITH_ARABIC.search(line)
            if m:
                po_num = m.group(1)
                out.append(("po_reference (direct)", po_num))
                if debug:
                    st.write(f"DEBUG: Found PO Reference directly in line: '{line}' -> {po_num}")
                continue
                
            # Fallback to original method
            idx = low.find("po reference")
            after = line[idx+len("po reference"):].strip() if idx != -1 else ""
            block = collect_block(lines, i, max_lines=10)
            scope = norm_advanced((after + " " + block).strip())

            if "pending" in scope.lower():
                continue

            m = RX_PO_PREFIX.search(scope)
            if m:
                a, b = m.group(1), m.group(2)
                out.append(("po_reference (prefix)", f"{a}/{b}" if b else a))
                continue

            m = RX_TWO_NUMS.search(scope)
            if m:
                out.append(("po_reference (bare+context)", f"{m.group(1)}/{m.group(2)}"))
                continue

            m = RX_SINGLE_NUM.search(scope)
            if m:
                out.append(("po_reference (bare+context)", m.group(1)))
    return out

def extract_from_description(lines, debug=False):
    """Extract PO from Description section - v48"""
    start_idx = None
    for i, raw in enumerate(lines):
        low = norm_advanced(raw).lower()
        if "code name" in low and "description" in low:
            start_idx = i
            break
        if low.strip() == "description":
            start_idx = i
            break
    
    if debug:
        if start_idx is None:
            st.write("DEBUG: No description section found")
        else:
            st.write(f"DEBUG: Found description section at line {start_idx}: '{lines[start_idx]}'")
    
    if start_idx is None:
        return None

    # Collect a much larger block to capture more content
    block = collect_block(lines, start_idx, max_lines=500)
    if debug:
        st.write(f"DEBUG: Raw block collected ({len(block.split() if block else [])} words): '{block[:200]}...'")
    if not block:
        if debug:
            st.write("DEBUG: No block content collected")
        return None
    
    # Also search the raw text around this area for PO patterns
    extended_search_start = max(0, start_idx - 50)
    extended_search_end = min(len(lines), start_idx + 500)
    extended_lines = lines[extended_search_start:extended_search_end]
    extended_text = "\n".join(extended_lines)
    
    if debug:
        st.write(f"DEBUG: Extended search area ({extended_search_start}-{extended_search_end}):\n{extended_text[:1000]}...")

    # Search in both the focused block and the extended area
    search_areas = [
        ("block", block),
        ("block_joined", " ".join(line.strip() for line in block.split('\n') if line.strip())), 
        ("extended", extended_text)
    ]

    # 1) Check for "PO NUM:" pattern first (highest priority)
    for area_name, search_text in search_areas:
        m = RX_PO_NUM_COLON.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found PO NUM: pattern in {area_name}: {m.group(1)}")
            return ("description (po-num-colon)", m.group(1))

    # 2) Check for "po/" pattern 
    for area_name, search_text in search_areas:
        m = RX_PO_FORWARD_SLASH.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found po/ pattern in {area_name}: {m.group(1)}")
            return ("description (po-forward-slash)", m.group(1))

    # 3) Check for Arabic "وجبات غذائيه" + number pattern
    for area_name, search_text in search_areas:
        m = RX_ARABIC_MEALS_NUM.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found Arabic meals number pattern in {area_name}: {m.group(1)}")
            return ("description (arabic-meals)", m.group(1))

    # 4) Check for concatenated "number+date" pattern
    for area_name, search_text in search_areas:
        m = RX_NUM_CONCAT_DATE.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found concatenated number-date pattern in {area_name}: {m.group(1)}")
            return ("description (num-concat-date)", m.group(1))

    # 5) Check for "number date" pattern
    for area_name, search_text in search_areas:
        m = RX_NUM_DATE.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found number-date pattern in {area_name}: {m.group(1)}")
            return ("description (num-date)", m.group(1))

    # 6) Check for standalone 6-digit numbers at end of blocks
    for area_name, search_text in search_areas:
        m = RX_STANDALONE_6DIGIT.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found standalone 6-digit number in {area_name}: {m.group(1)}")
            return ("description (standalone-6digit)", m.group(1))

    # 7) Check for "PO/number" pattern
    for area_name, search_text in search_areas:
        m = RX_PO_SLASH.search(search_text)
        if m:
            if debug:
                st.write(f"DEBUG: Found PO/number in {area_name}: {m.group(1)}")
            return ("description (po-slash)", m.group(1))

    # 8) "PO ..." patterns
    for area_name, search_text in search_areas:
        m = RX_PO_PREFIX.search(search_text)
        if m:
            a, b = m.group(1), m.group(2)
            return ("description (po-prefix)", f"{a}/{b}" if b else a)

    # 9) Direct anchored patterns
    for area_name, search_text in search_areas:
        for rx in (RX_CODE_ANCHOR_LINE, RX_CODE_SLASH_NUM, RX_CODE_NO_SPACE, RX_BOOKING_ORDER, RX_ARABIC_D_PREFIX, RX_ARABIC_D_SUFFIX):
            m = rx.search(search_text)
            if m:
                return ("description (anchored)", m.group(1))

    # 10) Numbers near anchor words
    for area_name, search_text in search_areas:
        num = (find_num_after_anchor_extended(search_text, 500) or
               find_num_after_anchor(search_text, ANCHOR_WORDS, 400) or
               find_num_after_anchor(search_text, ORDER_WORDS, 400) or
               find_num_before_anchor(search_text, ANCHOR_WORDS, 150) or
               find_num_before_anchor(search_text, ORDER_WORDS, 150))
        if num:
            return ("description (anchored-near)", num)

    # 11) Description fallback
    return extract_from_description_fallback(lines, start_idx, debug)

def extract_from_description_fallback(lines, desc_start_idx, debug=False):
    """Continue searching for كود/ patterns in the description section - v48"""
    desc_end_idx = desc_start_idx + 1
    while desc_end_idx < len(lines):
        low = norm_advanced(lines[desc_end_idx]).lower()
        if is_label_line(low):
            break
        desc_end_idx += 1
    
    extended_block = []
    search_idx = desc_end_idx
    kod_pattern = re.compile(r"كود/", re.IGNORECASE)
    
    stop_tokens = ("taxpayer name", "total sales", "total amount", "taxpayer activity code", "signed by")
    max_search_lines = min(len(lines), search_idx + 100)
    
    for i in range(search_idx, max_search_lines):
        line = norm_advanced(lines[i])
        low = line.lower()
        
        if any(tok in low for tok in stop_tokens):
            break
            
        extended_block.append(line)
        
        if kod_pattern.search(line):
            extended_text = "\n".join(extended_block)
            if debug:
                st.write(f"DEBUG: Description fallback found كود/: {extended_text[:500]}...")
            
            # Try all extraction methods on this extended block
            # 1) Check for "PO/number" pattern first
            m = RX_PO_SLASH.search(extended_text)
            if m:
                return ("description-fallback (po-slash)", m.group(1))

            # 2) "PO ..." patterns
            m = RX_PO_PREFIX.search(extended_text)
            if m:
                a, b = m.group(1), m.group(2)
                return ("description-fallback (po-prefix)", f"{a}/{b}" if b else a)

            # 3) Direct anchored patterns
            for rx in (RX_CODE_ANCHOR_LINE, RX_CODE_SLASH_NUM, RX_CODE_NO_SPACE, RX_BOOKING_ORDER, RX_ARABIC_D_PREFIX, RX_ARABIC_D_SUFFIX):
                m = rx.search(extended_text)
                if m:
                    return ("description-fallback (anchored)", m.group(1))

            # 4) Numbers near anchor words
            num = (find_num_after_anchor_extended(extended_text, 500) or
                   find_num_after_anchor(extended_text, ANCHOR_WORDS, 400) or
                   find_num_after_anchor(extended_text, ORDER_WORDS, 400) or
                   find_num_before_anchor(extended_text, ANCHOR_WORDS, 150) or
                   find_num_before_anchor(extended_text, ORDER_WORDS, 150))
            if num:
                return ("description-fallback (anchored-near)", num)
    
    return None

def extract_from_to_block(lines, debug=False):
    """Extract PO from Recipients (To) section - v48"""
    # Find Recipients (To) section more robustly
    to_start_idx = None
    for i, raw in enumerate(lines):
        low = norm_advanced(raw).lower()
        if "recipients (to)" in low or "recipients to" in low:
            to_start_idx = i
            break
    
    if to_start_idx is None:
        return None
    
    # Collect the TO block with more lines
    to_block = collect_block(lines, to_start_idx, max_lines=100)
    scope = to_block
    
    # Also try the slice_between method as fallback
    if not scope:
        scope = slice_between(lines, "recipients (to)") or slice_between(lines, "recipients to")
    
    if not scope:
        return None
    
    if debug:
        st.write(f"DEBUG: Recipients To scope: {scope[:500]}...")

    # Check for "Taxpayer Name: number" pattern first
    m = RX_TAXPAYER_NAME_NUM.search(scope)
    if m:
        if debug:
            st.write(f"DEBUG: Found taxpayer name number in TO scope: {m.group(1)}")
        return ("to_block (taxpayer-name-num)", m.group(1))

    # Check individual lines for more specific patterns
    for line in scope.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        m = RX_TAXPAYER_NAME_NUM.search(line)
        if m:
            if debug:
                st.write(f"DEBUG: Found taxpayer name number in TO line: '{line}' -> {m.group(1)}")
            return ("to_block (taxpayer-name-num)", m.group(1))

    # Check for numbers in parentheses
    for line in scope.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try parentheses at the end first
        m = RX_PARENTHESES_NUM_END.search(line)
        if m:
            if debug:
                st.write(f"DEBUG: Found parentheses-end pattern in TO: '{line}' -> {m.group(1)}")
            return ("to_block (parentheses-end)", m.group(1))
        # Try any parentheses
        m = RX_PARENTHESES_NUM.search(line)
        if m:
            if debug:
                st.write(f"DEBUG: Found parentheses pattern in TO: '{line}' -> {m.group(1)}")
            return ("to_block (parentheses)", m.group(1))

    m = RX_PO_PREFIX.search(scope)
    if m:
        a, b = m.group(1), m.group(2)
        return ("to_block (po-prefix)", f"{a}/{b}" if b else a)

    for rx in (RX_CODE_ANCHOR_LINE, RX_CODE_SLASH_NUM, RX_CODE_NO_SPACE, RX_BOOKING_ORDER, RX_ARABIC_D_PREFIX, RX_ARABIC_D_SUFFIX):
        m = rx.search(scope)
        if m:
            return ("to_block (anchored)", m.group(1))

    num = (find_num_after_anchor(scope, ANCHOR_WORDS, 400) or
           find_num_after_anchor(scope, ORDER_WORDS, 400) or
           find_num_before_anchor(scope, ANCHOR_WORDS, 150) or
           find_num_before_anchor(scope, ORDER_WORDS, 150))
    if num:
        return ("to_block (anchored-near)", num)

    return None

def extract_global_patterns(text, lines, debug=False):
    """Search for PO patterns anywhere in the document as a fallback method - v48"""
    if debug:
        st.write("DEBUG: Running global pattern search...")
        st.write(f"DEBUG: Full text preview (first 2000 chars):\n{text[:2000]}...")
    
    # 1) Look for "PO NUM:" pattern anywhere
    m = RX_PO_NUM_COLON.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global PO NUM: pattern: {m.group(1)}")
        return ("global (po-num-colon)", m.group(1))
    
    # 2) Look for "po/" pattern anywhere  
    m = RX_PO_FORWARD_SLASH.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global po/ pattern: {m.group(1)}")
        return ("global (po-forward-slash)", m.group(1))
    
    # 3) Look for Arabic "وجبات غذائيه" + number pattern
    m = RX_ARABIC_MEALS_NUM.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global Arabic meals number: {m.group(1)}")
        return ("global (arabic-meals)", m.group(1))
    
    # Also try a more flexible Arabic pattern as backup
    backup_arabic = re.compile(r"وجبات.*?(\d{6})", re.IGNORECASE | re.DOTALL)
    m = backup_arabic.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global backup Arabic pattern: {m.group(1)}")
        return ("global (arabic-backup)", m.group(1))
    
    # 2) Look for concatenated "number+date" pattern
    m = RX_NUM_CONCAT_DATE.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global concatenated number-date: {m.group(1)}")
        return ("global (num-concat-date)", m.group(1))
    
    # 3) Look for "number date" pattern anywhere
    m = RX_NUM_DATE.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global number-date: {m.group(1)}")
        return ("global (num-date)", m.group(1))
    
    # 4) Look for standalone 6-digit numbers
    for line in lines:
        line_norm = norm_advanced(line).strip()
        if not line_norm:
            continue
        m = RX_STANDALONE_6DIGIT.search(line_norm)
        if m:
            # Make sure this isn't part of a price, tax, or other financial data
            low = line_norm.lower()
            if any(skip in low for skip in ["total", "amount", "price", "tax", "egp", "discount"]):
                continue
            if debug:
                st.write(f"DEBUG: Found global standalone 6-digit: '{line_norm}' -> {m.group(1)}")
            return ("global (standalone-6digit)", m.group(1))
    
    # 5) Look for "PO/number" anywhere in the document
    m = RX_PO_SLASH.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global PO/number: {m.group(1)}")
        return ("global (po-slash)", m.group(1))
    
    # 6) Look for "Taxpayer Name: number" anywhere
    m = RX_TAXPAYER_NAME_NUM.search(text)
    if m:
        if debug:
            st.write(f"DEBUG: Found global taxpayer name number: {m.group(1)}")
        return ("global (taxpayer-name-num)", m.group(1))
    
    # 7) Look for numbers in parentheses in company names
    for line in lines:
        line_norm = norm_advanced(line)
        # Skip lines that are clearly not company names
        low = line_norm.lower()
        if any(skip in low for skip in ["total", "amount", "price", "quantity", "tax", "code name", "item code"]):
            continue
            
        # Look for parentheses patterns
        m = RX_PARENTHESES_NUM_END.search(line_norm)
        if m:
            if debug:
                st.write(f"DEBUG: Found global parentheses-end: '{line_norm}' -> {m.group(1)}")
            return ("global (parentheses-end)", m.group(1))
        
        m = RX_PARENTHESES_NUM.search(line_norm)
        if m:
            # Additional check: make sure this looks like a company name context
            if any(indicator in low for indicator in ["name", "taxpayer", "مكتب", "شركة", "فارما"]):
                if debug:
                    st.write(f"DEBUG: Found global parentheses in name context: '{line_norm}' -> {m.group(1)}")
                return ("global (parentheses)", m.group(1))
    
    return None

# v48 Priority list - updated to match standalone script exactly
PRIORITY = [
    "po_reference (direct)",           # Direct from PO Reference line
    "description (po-num-colon)",      # PO NUM: pattern - highest priority
    "global (po-num-colon)",           # PO NUM: anywhere
    "description (p-space-o)",         # p o number pattern
    "global (p-space-o)",              # p o number anywhere  
    "description (num-before-kod-slash)", # number/ كود pattern
    "global (num-before-kod-slash)",   # number/ كود anywhere
    "description (num-before-ready-meals)", # number وجبات جاهزة pattern
    "global (num-before-ready-meals)", # number وجبات جاهزة anywhere
    "description (po-forward-slash)",  # po/number pattern  
    "global (po-forward-slash)",       # po/number anywhere
    "description (arabic-meals)",      # Arabic "وجبات غذائيه" + number
    "global (arabic-meals)",           # Arabic "وجبات غذائيه" + number anywhere
    "global (arabic-backup)",          # Backup Arabic pattern
    "description (num-concat-date)",   # Concatenated number+date
    "global (num-concat-date)",        # Concatenated number+date anywhere
    "description (standalone-6digit)", # Standalone 6-digit at end  
    "global (standalone-6digit)",      # Standalone 6-digit anywhere
    "description (num-date)",          # Number followed by date
    "global (num-date)",               # Number followed by date anywhere
    "description (po-slash)",          # PO/number in description
    "global (po-slash)",               # PO/number anywhere
    "to_block (taxpayer-name-num)",    # Taxpayer Name: number
    "global (taxpayer-name-num)",      # Taxpayer Name: number anywhere
    "to_block (parentheses-end)",      # Numbers in parentheses at end
    "global (parentheses-end)",        # Numbers in parentheses at end anywhere
    "to_block (parentheses)",          # Numbers in parentheses
    "global (parentheses)",            # Numbers in parentheses anywhere
    "po_reference (prefix)",
    "po_reference (bare+context)",
    "description (po-prefix)",
    "description (anchored)",
    "description (anchored-near)",
    "description-fallback (po-slash)", 
    "description-fallback (po-prefix)",
    "description-fallback (anchored)",
    "description-fallback (anchored-near)",
    "to_block (po-prefix)",
    "to_block (anchored)",
    "to_block (anchored-near)",
]

def choose_best(candidates):
    """Choose the best PO candidate based on priority - v48"""
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: PRIORITY.index(x[0]) if x[0] in PRIORITY else 999)[0]

def advanced_po_extraction_v48(text: str, debug=False) -> tuple:
    """Main PO extraction function using v48 advanced logic"""
    if not text:
        return "", "", "no PO found"
    
    normalized_text = norm_advanced(text)
    lines = [norm_advanced(l) for l in normalized_text.splitlines()]

    # Collect forbidden numbers (Taxpayer Activity Code)
    forbidden = collect_forbidden_numbers(normalized_text)

    candidates = []
    candidates += extract_from_po_reference(lines, debug=debug)
    
    desc_result = extract_from_description(lines, debug=debug)
    if desc_result:
        candidates.append(desc_result)
        
    to_result = extract_from_to_block(lines, debug=debug)
    if to_result:
        candidates.append(to_result)
    
    # NEW: Add global search as fallback
    global_result = extract_global_patterns(normalized_text, lines, debug=debug)
    if global_result:
        candidates.append(global_result)

    # Filter out forbidden numbers
    candidates = [c for c in candidates if c and c[1] not in forbidden]

    best = choose_best(candidates)
    if not best:
        return "", "", "no PO found"
    
    src, val = best
    if isinstance(val, str) and "pending" in val.lower():
        return "", "", "no PO found"
    
    return val, src, ""

# ================= HELPER FUNCTIONS =================

ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
RLM = "\u200F"

def norm_num(s: str) -> str:
    """Normalize Arabic numbers to English and clean up"""
    return str(s or "").translate(ARABIC_NUM_MAP).replace("\u200f", "").replace("\u200e", "").strip()

def ensure_rtl(s: str) -> str:
    """Ensure RTL marker for Arabic text"""
    return (RLM + s) if s and ARABIC_RE.search(s) else s

def shape_ar(s: str) -> str:
    """Shape Arabic text properly"""
    if not s: return s
    if AR_FIX_OK and ARABIC_RE.search(s):
        try:
            return get_display(arabic_reshaper.reshape(s))
        except Exception: pass
    parts = s.split()
    fixed = [(p[::-1] if ARABIC_RE.search(p) and not re.search(r"[A-Za-z0-9]", p) else p) for p in parts]
    return " ".join(fixed)

def to_date_only(s: str) -> str:
    """Convert date string to YYYY-MM-DD format only"""
    if not s:
        return ""
    try:
        cleaned_s = s.strip()
        cleaned_s = re.sub(r"\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M.*", "", cleaned_s, flags=re.IGNORECASE)
        cleaned_s = re.sub(r"\s+\(\d{1,2}/\d{1,2}/\d{4}.*\)", "", cleaned_s)

        dt = pd.to_datetime(cleaned_s, dayfirst=True, errors="coerce")
        if pd.isna(dt): 
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                return s.strip()
        return dt.strftime("%Y-%m-%d")
    except Exception: 
        return s.strip()

def extract_text(file_content: bytes) -> str:
    """Extract text from PDF bytes"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_content))
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
    except Exception: pass
    return text

def one(m): 
    """Helper to extract first group from regex match"""
    return m.group(1).strip() if m else ""

def extract_document_type(text: str) -> str:
    """Extract document type from text instead of hardcoding 'Invoice'"""
    type_patterns = [
        r"Document\s*Type\s*:\s*([A-Za-z\s]+)",
        r"Type\s*:\s*([A-Za-z\s]+)",
        r"\b(Invoice|Credit\s*Note|Debit\s*Note|Receipt)\b",
        r"(فاتورة|إشعار\s*دائن|إشعار\s*مدين)"
    ]
    
    for pattern in type_patterns:
        m = re.search(pattern, text, re.I)
        if m:
            doc_type = m.group(1).strip()
            if "credit" in doc_type.lower():
                return "Credit Note"
            elif "debit" in doc_type.lower():
                return "Debit Note"
            elif "invoice" in doc_type.lower() or "فاتورة" in doc_type:
                return "Invoice"
            elif "receipt" in doc_type.lower():
                return "Receipt"
            return doc_type.title()
    
    if re.search(r"credit\s*note", text, re.I):
        return "Credit Note"
    
    return "Invoice"

def extract_from_name(text: str, ocr_text_val: str = "") -> str:
    """Extract taxpayer name from document"""
    s = re.search(r"Issuer\s*\(\s*From\s*\)", text, re.I)
    block = text[s.end():] if s else text
    e = re.search(r"(Recipients\s*\(\s*To\s*\)|\bID\s*:|Proforma\s*Invoice\s*Number|SO\s*Reference|Registration\s*Number)", block, re.I)
    if e: block = block[:e.start()]
    m = re.search(r"Taxpayer\s*Name\s*[:：]\s*([^\n]+)", block, re.I)
    name_lines = []
    if m:
        rest = m.group(1)
        lines = [rest] + block[m.end():].splitlines()
        for ln in lines:
            ln = ln.strip()
            if not ln: continue
            if re.search(r"Registration\s*Number", ln, re.I): break
            if re.search(r"\b(EG|Cairo|Street|Road|St\.?)\b", ln, re.I): break
            name_lines.append(ln)
    name = " ".join([x for x in name_lines if x not in ["(", ")"]]).strip()
    if (not name or len(name) <= 2) and ocr_text_val:
        return extract_from_name(ocr_text_val, ocr_text_val="")
    return ensure_rtl(shape_ar(name))

def parse_pdf_fields(file_content: bytes, filename: str, debug=False) -> dict:
    """Parse all fields from a PDF file using v48 PO extraction"""
    try:
        base_text = extract_text(file_content)
        T = base_text
        fields = {"filename": filename}
        
        fields["STATUS"] = one(re.search(r"Status\s*:\s*([A-Za-z]+)", T))
        fields["Submission Date"] = one(re.search(r"Submission Date\s*:\s*([^\n]+)", T))
        fields["Issuance Date"] = one(re.search(r"Issuance Date\s*:\s*([^\n]+)", T))
        fields["ID"] = one(re.search(r"\bID\s*:\s*([A-Z0-9]+)", T))
        fields["Taxpayer Name"] = extract_from_name(T)
        fields["Issuer Registration Number"] = one(re.search(r"Issuer\s*\(From\)[\s\S]*?Registration Number\s*:\s*#?([0-9]+)", T))
        
        regs = re.findall(r"Registration Number\s*:\s*#?([0-9]+)", T)
        fields["Recipient Registration Number"] = regs[1] if len(regs) >= 2 else ""
        
        fields["Total Amount (EGP)"] = one(re.search(r"Total Amount\s*\(EGP\)\s*([0-9\.,]+)", T))
        fields["Internal ID Raw"] = one(re.search(r"Internal\s*ID\s*:\s*([^\n]+)", T, re.I))
        fields["Document Type"] = extract_document_type(T)
        
        # Use v48 PO extraction
        po_number, po_source, po_note = advanced_po_extraction_v48(T, debug=debug)
        fields["PO number"] = po_number
        fields["PO source"] = po_source
        fields["PO note"] = po_note
        
        return fields
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return {"filename": filename, "error": str(e)}

# ================= STREAMLIT UI =================

def main():
    st.title("PDF Data Extractor v48")
    st.markdown("Extract structured data from PDF files with **advanced v48 PO number detection**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Template upload
    st.sidebar.subheader("Excel Template (Optional)")
    template_file = st.sidebar.file_uploader(
        "Upload Excel template to define column schema",
        type=['xlsx', 'xls'],
        help="Upload an Excel file to use its columns as the output schema"
    )
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    max_workers = st.sidebar.slider(
        "Number of parallel workers",
        min_value=1,
        max_value=min(8, os.cpu_count()),
        value=max(2, os.cpu_count() - 1),
        help="More workers = faster processing, but uses more memory"
    )
    
    debug_mode = st.sidebar.checkbox(
        "Enable debug mode",
        value=False,
        help="Show detailed PO extraction debugging information"
    )
    
    # Arabic text fixing options
    st.sidebar.subheader("Arabic Text Enhancement")
    
    use_llm_arabic = st.sidebar.checkbox(
        "Use LLM for Arabic text fixing",
        value=False,
        help="Use AI to fix garbled Arabic company names (requires API key or local model)"
    )
    
    if use_llm_arabic:
        llm_method = st.sidebar.selectbox(
            "LLM Method",
            options=["OpenAI API", "Local Ollama", "Both (fallback)"],
            index=2,
            help="Choose how to fix Arabic text"
        )
        
        if llm_method in ["OpenAI API", "Both (fallback)"]:
            openai_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for Arabic text fixing"
            )
            if openai_key:
                st.session_state.openai_api_key = openai_key
        
        if llm_method in ["Local Ollama", "Both (fallback)"]:
            st.sidebar.info("Make sure Ollama is running locally with llama3.2:1b model")
    
    use_ocr = st.sidebar.checkbox(
        "Enable OCR fallback",
        value=False,
        help="Use OCR if PDF text extraction fails (slower but more accurate for scanned PDFs)"
    )
    
    if use_ocr and not TESS_OK:
        st.sidebar.warning("OCR libraries not available. Please install pytesseract and pdf2image.")
        use_ocr = False
    
    # Main interface
    st.header("Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files to process",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )
    
    # Load template schema if provided
    schema_cols = None
    if template_file is not None:
        try:
            xls = pd.ExcelFile(template_file)
            schema_cols = list(xls.parse(xls.sheet_names[0]).columns.astype(str))
            st.success(f"Template loaded with {len(schema_cols)} columns: {', '.join(schema_cols[:5])}{'...' if len(schema_cols) > 5 else ''}")
        except Exception as e:
            st.error(f"Error loading template: {str(e)}")
            schema_cols = None
    
    # Default schema if no template provided
    if schema_cols is None:
        schema_cols = [
            "INTERNAL ID -1", "INTERNAL ID -2", "DATE", "TYPE", "version",
            "TOTAL VALUE EGP", "FROM", "REGESTRAION NUMBER", "STATUS",
            "REGESTRAION", "PO number", "PO Reference"
        ]
        st.info(f"Using default schema with {len(schema_cols)} columns")
    
    # Process files when uploaded
    if uploaded_files:
        st.header("Processing Results")
        
        # Show file info
        total_size = sum(len(f.read()) for f in uploaded_files)
        for f in uploaded_files:  # Reset file pointers
            f.seek(0)
        
        st.info(f"Ready to process {len(uploaded_files)} files ({total_size / 1024 / 1024:.1f} MB total)")
        
        if st.button("Start Processing", type="primary"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Debug output container
            if debug_mode:
                debug_container = st.container()
                debug_expander = debug_container.expander("Debug Output", expanded=True)
            
            # Process files
            rows = []
            start_time = time.time()
            
            # Process files sequentially for Streamlit
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                progress_bar.progress((i) / len(uploaded_files))
                
                # Read file content
                file_content = uploaded_file.read()
                
                # Show debug info for this file
                if debug_mode:
                    with debug_expander:
                        st.subheader(f"Debug: {uploaded_file.name}")
                        
                # Parse fields with debug output
                if debug_mode:
                    with debug_expander:
                        fields = parse_pdf_fields(file_content, uploaded_file.name, debug=True)
                else:
                    fields = parse_pdf_fields(file_content, uploaded_file.name, debug=False)
                
                # Create row mapping
                date_value = fields.get("Submission Date", "") or fields.get("Issuance Date", "")
                
                mapping = {
                    "INTERNAL ID -1": fields.get("ID", ""),
                    "INTERNAL ID -2": fields.get("Internal ID Raw", ""),
                    "DATE": to_date_only(date_value),
                    "TYPE": fields.get("Document Type", "Invoice"),
                    "version": "",
                    "TOTAL VALUE EGP": fields.get("Total Amount (EGP)", ""),
                    "FROM": fields.get("Taxpayer Name", ""),
                    "REGESTRAION NUMBER": fields.get("Issuer Registration Number", ""),
                    "STATUS": fields.get("STATUS", ""),
                    "REGESTRAION": fields.get("Recipient Registration Number", ""),
                    "PO number": fields.get("PO number", ""),
                    "PO Reference": fields.get("PO number", ""),
                    "PO source": fields.get("PO source", ""),
                    "filename": uploaded_file.name
                }
                
                row = {col: mapping.get(col, "") for col in schema_cols}
                if "PO number" not in schema_cols:
                    row["PO number"] = fields.get("PO number", "")
                if "PO source" not in schema_cols:
                    row["PO source"] = fields.get("PO source", "")
                if "filename" not in schema_cols:
                    row["filename"] = uploaded_file.name
                
                rows.append(row)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Complete processing
            end_time = time.time()
            processing_time = end_time - start_time
            
            status_text.text("Processing completed!")
            progress_bar.progress(1.0)
            
            # Create output DataFrame
            output_cols = list(schema_cols)
            if "PO number" not in output_cols:
                output_cols.append("PO number")
            if "PO source" not in output_cols:
                output_cols.append("PO source")
            if "filename" not in output_cols:
                output_cols.append("filename")
            
            df = pd.DataFrame(rows, columns=output_cols)
            
            # Show results
            st.success(f"Successfully processed {len(rows)} files in {processing_time:.1f} seconds")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(rows))
            with col2:
                po_found = sum(1 for row in rows if row.get("PO number", ""))
                st.metric("PO Numbers Found", po_found)
            with col3:
                success_rate = (po_found / len(rows) * 100) if rows else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                avg_time = processing_time / len(rows) if rows else 0
                st.metric("Avg Time/File", f"{avg_time:.2f}s")
            
            # PO Source Distribution
            if "PO source" in df.columns:
                po_sources = df[df['PO number'].str.strip() != '']['PO source'].value_counts()
                if not po_sources.empty:
                    st.subheader("PO Extraction Methods Used")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(po_sources)
                    with col2:
                        st.write("**Method Distribution:**")
                        for method, count in po_sources.items():
                            st.write(f"• {method}: {count} files")
            
            # Display results table
            st.subheader("Extracted Data")
            
            # Add filters
            with st.expander("Filter Results"):
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    # Filter by document type
                    doc_types = df['TYPE'].unique().tolist() if 'TYPE' in df.columns else []
                    selected_types = st.multiselect(
                        "Filter by Document Type",
                        options=doc_types,
                        default=doc_types
                    )
                
                with filter_col2:
                    # Filter by PO number presence
                    po_filter = st.selectbox(
                        "Filter by PO Number",
                        options=["All", "With PO Number", "Without PO Number"],
                        index=0
                    )
            
            # Apply filters
            filtered_df = df.copy()
            
            if 'TYPE' in filtered_df.columns and selected_types:
                filtered_df = filtered_df[filtered_df['TYPE'].isin(selected_types)]
            
            if po_filter == "With PO Number":
                filtered_df = filtered_df[filtered_df['PO number'].str.strip() != ""]
            elif po_filter == "Without PO Number":
                filtered_df = filtered_df[filtered_df['PO number'].str.strip() == ""]
            
            # Display filtered results
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.info(f"Showing {len(filtered_df)} of {len(df)} records")
            
            # Download options
            st.subheader("Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel download
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Extracted_Data')
                excel_buffer.seek(0)
                
                st.download_button(
                    label="Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"pdf_extraction_results_v48_{int(time.time())}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # CSV download
                csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download as CSV",
                    data=csv_buffer,
                    file_name=f"pdf_extraction_results_v48_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
            # Detailed analysis
            if st.checkbox("Show Detailed Analysis"):
                st.subheader("Analysis")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    if 'TYPE' in df.columns:
                        st.write("**Document Types Distribution:**")
                        type_counts = df['TYPE'].value_counts()
                        st.bar_chart(type_counts)
                
                with analysis_col2:
                    if 'PO number' in df.columns:
                        st.write("**PO Number Extraction Status:**")
                        po_status = df['PO number'].apply(lambda x: 'Found' if str(x).strip() else 'Not Found').value_counts()
                        st.bar_chart(po_status)
                
                # Show files with issues
                issues_df = df[df['PO number'].str.strip() == ""]
                if not issues_df.empty:
                    with st.expander(f"Files without PO Numbers ({len(issues_df)} files)"):
                        st.dataframe(
                            issues_df[['filename'] + [col for col in ['FROM', 'TYPE', 'DATE'] if col in issues_df.columns]],
                            use_container_width=True,
                            hide_index=True
                        )

    # Information section
    st.header("Information")
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Upload Template (Optional)**: Upload an Excel file to define the output column schema
        2. **Configure Settings**: Adjust parallel workers, enable debug mode, and OCR if needed
        3. **Upload PDF Files**: Select one or more PDF files to process
        4. **Process**: Click "Start Processing" to extract data
        5. **Review Results**: View extracted data in the table below
        6. **Download**: Export results as Excel or CSV files
        
        **Supported Data Fields:**
        - Internal IDs and document identifiers
        - Dates (Submission/Issuance)
        - Document types (Invoice, Credit Note, etc.)
        - Total amounts in EGP
        - Taxpayer names and registration numbers
        - **PO Numbers** (with v48 advanced extraction)
        """)
    
    with st.expander("v48 Advanced PO Extraction Features"):
        st.markdown("""
        The app uses **v48** algorithms to extract PO numbers from various sources:
        
        **Extraction Sources (in priority order):**
        1. **PO Reference Section** - Direct PO references with Arabic text support
        2. **PO NUM: Format** - Numbers after "PO NUM:" or "PO NUM :" - NEW
        3. **po/ Format** - Numbers after "po/" (lowercase) - NEW  
        4. **Arabic Meals Pattern** - Numbers after "وجبات غذائيه" or "وجبات غذائية"  
        5. **Concatenated Dates** - Numbers combined with dates (e.g., "17418015/8/2025")
        6. **Standalone Numbers** - 6-digit numbers at end of descriptions
        7. **Number-Date Patterns** - Numbers followed by dates
        8. **PO/Number Format** - "PO/173822" patterns (uppercase)
        9. **Taxpayer Names** - Numbers in company names or after "Taxpayer Name:"
        10. **Parentheses Numbers** - Numbers in parentheses in company names
        11. **Description Section** - PO numbers in item descriptions
        12. **Recipients Section** - PO numbers in recipient information
        
        **Supported Formats:**
        - `PO NUM : 7901` or `PO NUM: 7906` (NEW - highest priority)
        - `po/172237` (NEW - lowercase slash format)
        - `PO Reference: 171525ثرى` (with Arabic text)
        - `وجبات غذائية د. 173417` (Arabic meals pattern)
        - `17418015/8/2025` (concatenated with date)
        - `PO/173822` (uppercase slash format)
        - `172829 1/8/2025` (number space date)
        - `173128` (standalone at end)
        - `Taxpayer Name: 26141`
        - `(مكتب علمي ام ام فارما (26141` (in parentheses)
        - `كود 1234` (Arabic code)
        - `Order Number: 1234`
        
        **Smart Features:**
        - Filters out taxpayer activity codes
        - Handles Arabic text normalization and variations
        - Supports RTL text direction
        - Prioritizes high-confidence matches
        - Global fallback search when section-specific extraction fails
        - Comprehensive pattern coverage for all discovered formats
        """)
    
    with st.expander("Technical Requirements"):
        st.markdown(f"""
        **System Status:**
        - PDF Processing: Available
        - {'Arabic Text Support: Available' if AR_FIX_OK else 'Arabic Text Support: Limited (install arabic-reshaper and python-bidi)'}
        - {'OCR Support: Available' if TESS_OK else 'OCR Support: Not Available (install pytesseract and pdf2image)'}
        
        **Installation:**
        ```bash
        pip install streamlit pandas PyPDF2 openpyxl
        pip install arabic-reshaper python-bidi  # For Arabic support
        pip install pytesseract pdf2image        # For OCR support
        ```
        
        **v48 Features:**
        - Complete pattern coverage for all discovered PO formats
        - Advanced Arabic text processing
        - Global search fallback mechanism
        - Detailed debug output for troubleshooting
        - Priority-based extraction method selection
        """)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
