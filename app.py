import streamlit as st
import os
import re
import time
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from io import BytesIO
import traceback

# Configure page
st.set_page_config(
    page_title="PDF Invoice Processor",
    page_icon="📄",
    layout="wide"
)

# ---------- Optional Arabic shaping ----------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    AR_FIX_OK = True
except Exception:
    arabic_reshaper = None
    get_display = None
    AR_FIX_OK = False

# ---------- Optional OCR ----------
@st.cache_resource
def configure_tesseract():
    try:
        import pytesseract
        # Try common Windows path
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.name == "nt" and os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path
        return True
    except Exception:
        return False

def ocr_text(path: Path, max_pages: int = 2) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), first_page=1, last_page=max_pages, fmt="png", dpi=300)
        txts = [pytesseract.image_to_string(im, lang="ara+eng") for im in images]
        return "\n".join(txts)
    except Exception:
        return ""

# ---------- PDF text extraction ----------
def extract_text_pypdf2(path: Path) -> str:
    try:
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                text += (p.extract_text() or "") + "\n"
        return text
    except Exception:
        return ""

def extract_text_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except Exception:
        return ""

def extract_text(path: Path) -> str:
    t1 = extract_text_pypdf2(path)
    t2 = extract_text_pdfplumber(path)
    parts = [p for p in [t1, t2] if p.strip()]
    return "\n".join(parts)

# ---------- Utilities ----------
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
RLM = "\u200F"
PO_KEYWORDS = [
    r"PO\s*Reference", r"PO\s*Ref", r"P\.?O\.?\s*Ref",
    r"P\.?O\.?", r"Purchase\s*Order", r"أمر\s*شراء",
]

def norm_num(s: str) -> str:
    return str(s or "").translate(ARABIC_NUM_MAP).replace("\u200f", "").replace("\u200e", "").strip()

def ensure_rtl(s: str) -> str:
    return (RLM + s) if s and ARABIC_RE.search(s) else s

def shape_ar(s: str) -> str:
    if not s: return s
    if AR_FIX_OK and ARABIC_RE.search(s):
        try:
            return get_display(arabic_reshaper.reshape(s))
        except Exception:
            pass
    # fallback: reverse Arabic-only tokens
    parts = s.split()
    fixed = [(p[::-1] if ARABIC_RE.search(p) and not re.search(r"[A-Za-z0-9]", p) else p) for p in parts]
    return " ".join(fixed)

def to_date_only(s: str) -> str:
    if not s: return ""
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt): return s.strip()
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s.strip()

def one(m): return m.group(1).strip() if m else ""

def multi_pass_po(text: str) -> List[str]:
    T = text
    results: List[str] = []
    m = re.search(r"(?:PO\s*Reference|PO\s*Ref|P\.?O\.?\s*Ref)\s*[:\-]?\s*([A-Za-z0-9\/\-\_]+)", T, re.I)
    if m and m.group(1).strip():
        cand = norm_num(m.group(1))
        if re.search(r"\d", cand) and "proforma" not in cand.lower():
            results.append(cand)
    for kw in PO_KEYWORDS:
        for mm in re.finditer(kw + r"\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})", T, re.I):
            val = norm_num(mm.group(1))
            if re.search(r"\d", val) and "proforma" not in val.lower():
                results.append(val)
    for mm in re.finditer(r"\bPO\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})\b", T, re.I):
        val = norm_num(mm.group(1))
        if re.search(r"\d", val) and "proforma" not in val.lower():
            results.append(val)
    # dedupe preserve order
    seen=set(); out=[]
    for r in results:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def extract_from_name(text: str, ocr_text: str = "") -> str:
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
    if (not name or len(name) <= 2) and ocr_text:
        return extract_from_name(ocr_text, ocr_text="")
    return ensure_rtl(shape_ar(name))

def parse_invoice_fields(text: str, ocr_text: str = "") -> Dict[str, Any]:
    T = text; f: Dict[str, Any] = {}
    f["STATUS"] = one(re.search(r"Status\s*:\s*([A-Za-z]+)", T))
    f["Submission Date"] = one(re.search(r"Submission Date\s*:\s*([^\n]+)", T))
    f["Issuance Date"] = one(re.search(r"Issuance Date\s*:\s*([^\n]+)", T))
    f["ID"] = one(re.search(r"\bID\s*:\s*([A-Z0-9]+)", T))
    f["Taxpayer Name"] = extract_from_name(T, ocr_text=ocr_text)
    f["Issuer Registration Number"] = one(re.search(r"Issuer\s*\(From\)[\s\S]*?Registration Number\s*:\s*#?([0-9]+)", T))
    regs = re.findall(r"Registration Number\s*:\s*#?([0-9]+)", T)
    f["Recipient Registration Number"] = regs[1] if len(regs) >= 2 else ""
    f["Total Amount (EGP)"] = one(re.search(r"Total Amount\s*\(EGP\)\s*([0-9\.,]+)", T))
    f["Internal ID Raw"] = one(re.search(r"Internal\s*ID\s*:\s*([^\n]+)", T, re.I))
    po = multi_pass_po(T if T else "")
    if not po and ocr_text: po = multi_pass_po(ocr_text)
    f["PO number"] = po[0] if po else ""
    return f

def map_to_schema(fields: Dict[str, Any], schema_cols: List[str]) -> Dict[str, Any]:
    mapping = {
        "INTERNAL ID -1": fields.get("ID", ""),
        "INTERNAL ID -2": fields.get("Internal ID Raw", ""),
        "DATE": to_date_only(fields.get("Issuance Date", "")),
        "TYPE": "Invoice",
        "version": "",
        "TOTAL VALUE EGP": fields.get("Total Amount (EGP)", ""),
        "FROM": fields.get("Taxpayer Name", ""),
        "REGESTRAION NUMBER": fields.get("Issuer Registration Number", ""),
        "STATUS": fields.get("STATUS", ""),
        "REGESTRAION": fields.get("Recipient Registration Number", ""),
        "PO number": fields.get("PO number", ""),
        "PO Reference": fields.get("PO number", ""),
    }
    row = {col: mapping.get(col, "") for col in schema_cols}
    return row

# ---------- Worker function ----------
def process_pdf_worker(args: Tuple[str, List[str], bool]) -> Tuple[str, Dict[str, Any], str]:
    pdf_path, schema_cols, use_ocr = args
    try:
        p = Path(pdf_path)
        base_text = extract_text(p)
        ocr_result = ""
        if use_ocr:
            configure_tesseract()
            ocr_result = ocr_text(p, max_pages=2)
        
        fields = parse_invoice_fields(base_text, ocr_text=ocr_result)
        row = map_to_schema(fields, schema_cols)
        return (p.name, row, "")
    except Exception as e:
        error_msg = f"Error processing {Path(pdf_path).name}: {str(e)}"
        empty_row = {col: "" for col in schema_cols}
        return (Path(pdf_path).name, empty_row, error_msg)

# ---------- Main processing function ----------
def process_pdfs(pdf_files, schema_cols: List[str], max_workers: int = 4, use_ocr: bool = False):
    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    
    for pdf_file in pdf_files:
        temp_path = Path(temp_dir) / pdf_file.name
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        pdf_paths.append(str(temp_path))
    
    rows = []
    errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf_worker, (pdf_path, schema_cols, use_ocr)) 
                  for pdf_path in pdf_paths]
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            filename, row, error = future.result()
            rows.append(row)
            if error:
                errors.append(error)
            
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Processing: {completed}/{total} files completed")
    
    duration = time.time() - start_time
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    return rows, errors, duration

# ---------- Streamlit App ----------
def main():
    st.title("📄 PDF Invoice Processor")
    st.markdown("Upload PDF invoices to extract structured data with parallel processing")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # CPU cores configuration
    max_cpu_cores = multiprocessing.cpu_count()
    max_workers = st.sidebar.slider(
        "Number of Workers", 
        min_value=1, 
        max_value=max_cpu_cores, 
        value=min(4, max_cpu_cores),
        help=f"Number of parallel workers (CPU cores available: {max_cpu_cores})"
    )
    
    # OCR option
    use_ocr = st.sidebar.checkbox(
        "Enable OCR Fallback", 
        value=False,
        help="Use OCR for PDFs with poor text extraction (slower but more accurate)"
    )
    
    if use_ocr:
        if not configure_tesseract():
            st.sidebar.warning("⚠️ OCR enabled but Tesseract not found. Install Tesseract OCR for full functionality.")
    
    # Schema columns (predefined)
    schema_cols = [
        "INTERNAL ID -1", "INTERNAL ID -2", "DATE", "TYPE", "version", 
        "TOTAL VALUE EGP", "FROM", "REGESTRAION NUMBER", "STATUS", 
        "REGESTRAION", "PO number"
    ]
    
    st.sidebar.markdown("### 📋 Output Schema")
    st.sidebar.code(", ".join(schema_cols), language="text")
    
    # File upload
    st.header("📁 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Select one or more PDF invoice files to process"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF files uploaded")
        
        # Show file list
        with st.expander("View uploaded files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size:,} bytes)")
        
        # Process button
        if st.button("🚀 Process PDFs", type="primary"):
            st.header("⚡ Processing Results")
            
            with st.spinner("Processing PDFs..."):
                rows, errors, duration = process_pdfs(
                    uploaded_files, 
                    schema_cols, 
                    max_workers=max_workers, 
                    use_ocr=use_ocr
                )
            
            # Show processing stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Files Processed", len(rows))
            with col2:
                st.metric("Processing Time", f"{duration:.1f}s")
            with col3:
                st.metric("Workers Used", max_workers)
            with col4:
                st.metric("Errors", len(errors))
            
            # Show errors if any
            if errors:
                with st.expander("⚠️ Processing Errors", expanded=True):
                    for error in errors:
                        st.error(error)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=schema_cols)
            
            # Display results
            st.header("📊 Extracted Data")
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            st.header("📈 Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Completeness")
                completeness = {}
                for col in schema_cols:
                    non_empty = df[col].astype(str).str.strip().ne('').sum()
                    completeness[col] = f"{non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)"
                
                for col, stat in completeness.items():
                    st.write(f"**{col}:** {stat}")
            
            with col2:
                st.subheader("PO Number Analysis")
                po_with_data = df['PO number'].astype(str).str.strip().ne('').sum()
                st.metric("PDFs with PO Numbers", po_with_data, f"{po_with_data/len(df)*100:.1f}%")
                
                if po_with_data > 0:
                    st.write("**Sample PO Numbers:**")
                    sample_pos = df[df['PO number'].astype(str).str.strip().ne('')]['PO number'].head(5).tolist()
                    for po in sample_pos:
                        st.code(po)
            
            # Download options
            st.header("💾 Download Results")
            
            # Excel download
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Invoice_Data', index=False)
            
            st.download_button(
                label="📥 Download Excel File",
                data=buffer.getvalue(),
                file_name=f"invoice_data_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV File",
                data=csv,
                file_name=f"invoice_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Please upload PDF files to begin processing")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔧 Requirements
    - **PyPDF2** or **pdfplumber** for PDF text extraction
    - **pytesseract** and **pdf2image** for OCR (optional)
    - **arabic-reshaper** and **python-bidi** for Arabic text support
    
    ### 📝 Notes
    - Processing time depends on file size and number of workers
    - OCR fallback is slower but more accurate for scanned PDFs
    - Arabic text is automatically detected and properly formatted
    """)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows compatibility
    main()