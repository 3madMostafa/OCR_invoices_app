# PDF Data Extractor - Streamlit App

A powerful web application for extracting structured data from PDF files with advanced PO (Purchase Order) number detection, built with Streamlit.

## Features

### 🚀 Core Capabilities
- **Batch PDF Processing**: Upload and process multiple PDF files simultaneously
- **Advanced PO Extraction**: Intelligent extraction of PO numbers from various document sections
- **Arabic Text Support**: Full RTL text support with proper Arabic text shaping
- **Flexible Schema**: Use custom Excel templates to define output columns
- **Real-time Processing**: Live progress tracking and instant results
- **Multiple Export Formats**: Download results as Excel or CSV

### 🔍 Advanced PO Number Detection
The app uses sophisticated algorithms to extract PO numbers from:
- PO Reference sections
- Item descriptions
- Recipient information blocks
- Arabic text patterns (كود, أوردر رقم, etc.)
- Compound number formats (1234/5678)

### 📊 Data Fields Extracted
- Document IDs and internal references
- Submission and issuance dates
- Document types (Invoice, Credit Note, Debit Note)
- Total amounts in EGP
- Taxpayer names and registration numbers
- Purchase Order numbers (advanced detection)

## Installation

### 1. Clone or Download
Save the `app.py` file and `requirements.txt` to your project directory.

### 2. Install Dependencies

#### Basic Installation:
```bash
pip install -r requirements.txt
```

#### Or install individually:
```bash
# Core dependencies (required)
pip install streamlit pandas PyPDF2 openpyxl

# Arabic text support (recommended)
pip install arabic-reshaper python-bidi

# OCR support (optional, for scanned PDFs)
pip install pytesseract pdf2image Pillow
```

### 3. OCR Setup (Optional)
If you want OCR support for scanned PDFs:

#### Windows:
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. The app will auto-detect the installation

#### macOS:
```bash
brew install tesseract
```

#### Linux:
```bash
sudo apt-get install tesseract-ocr
```

## Usage

### 1. Start the Application
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### 2. Configure Settings (Sidebar)
- **Excel Template**: Upload an Excel file to define output columns (optional)
- **Parallel Workers**: Adjust based on your system (more workers = faster processing)
- **OCR Fallback**: Enable for scanned PDFs (requires Tesseract installation)

### 3. Process Files
1. **Upload PDFs**: Use the file uploader to select PDF files
2. **Start Processing**: Click the "Start Processing" button
3. **View Results**: Review extracted data in the interactive table
4. **Filter Results**: Use filters to focus on specific document types or PO status
5. **Download**: Export results as Excel or CSV files

### 4. Advanced Features
- **Filtering**: Filter by document type or PO number presence
- **Analysis**: View distribution charts and identify processing issues
- **Batch Processing**: Handle hundreds of files in a single session

## File Structure

```
pdf-extractor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Configuration Options

### Processing Settings
- **Workers**: 1-8 parallel workers (default: CPU cores - 1)
- **OCR**: Enable/disable OCR fallback for scanned documents
- **Template**: Custom Excel schema for output columns

### Default Output Schema
If no template is provided, the app uses these columns:
- INTERNAL ID -1, INTERNAL ID -2
- DATE, TYPE, version
- TOTAL VALUE EGP
- FROM, REGESTRAION NUMBER
- STATUS, REGESTRAION
- PO number, PO Reference

## Troubleshooting

### Common Issues

#### 1. Arabic Text Not Displaying Properly
```bash
pip install arabic-reshaper python-bidi
```

#### 2. OCR Not Working
- Install Tesseract OCR for your operating system
- For Windows, ensure it's installed in the default location
- Check if pytesseract and pdf2image are installed

#### 3. Memory Issues with Large Files
- Reduce the number of parallel workers
- Process files in smaller batches
- Close other applications to free up memory

#### 4. Slow Processing
- Disable OCR if not needed
- Increase the number of workers (up to your CPU cores)
- Use an SSD for better I/O performance

### Error Messages
- **"OCR libraries not available"**: Install pytesseract and pdf2image
- **"Error loading template"**: Check Excel file format and content
- **"Processing failed"**: Check PDF file integrity and permissions

## Performance Tips

### For Best Performance:
1. **Disable OCR** unless you have scanned PDFs
2. **Use optimal worker count** (usually CPU cores - 1)
3. **Process similar file types together**
4. **Use Excel templates** to avoid column mismatches
5. **Close other heavy applications** during processing

### Typical Processing Speed:
- **Text-based PDFs**: 1-3 seconds per file
- **With OCR**: 5-15 seconds per file
- **Batch processing**: Scales linearly with worker count

## Advanced Usage

### Custom Templates
Create an Excel file with your desired column headers and upload it as a template. The app will use these columns for the output structure.

### API Integration
The core processing functions can be used independently of Streamlit for integration into other applications:

```python
from app import parse_pdf_fields, advanced_po_extraction

# Process a single PDF
with open('document.pdf', 'rb') as f:
    fields = parse_pdf_fields(f.read(), 'document.pdf')

# Extract just PO numbers
with open('document.pdf', 'rb') as f:
    text = extract_text(f.read())
    po_number = advanced_po_extraction(text)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify PDF files are not corrupted or password-protected
4. Test with a small batch first before processing large quantities

## License

This project is provided as-is for educational and business use. Modify and adapt as needed for your specific requirements.
