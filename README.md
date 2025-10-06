# PDF to Markdown Converter using Phi-3 Vision

This script converts PDF documents to Markdown format using Microsoft's Phi-3.5 Vision Instruct model with 4-bit quantization for efficient inference.

## Features

- 📄 Extracts pages from PDF as high-resolution images
- 🤖 Uses Phi-3.5 Vision (quantized) for accurate OCR and structure detection
- 📝 Detects sections, subsections, and document hierarchy
- 💾 Saves individual page Markdown files and a combined document
- ⚡ 4-bit quantization for efficient memory usage
- 🎯 Preserves tables, lists, and formatting

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU)

## Installation

1. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

For macOS:
```bash
brew install poppler
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python pdf_to_markdown.py path/to/your/document.pdf
```

This will:
- Extract all pages from the PDF
- Convert each page to Markdown using Phi-3.5 Vision
- Save results to `results/` directory

### Advanced Usage

```bash
# Specify custom output directory
python pdf_to_markdown.py document.pdf --output-dir my_results

# Adjust image quality (higher DPI = better quality, slower processing)
python pdf_to_markdown.py document.pdf --dpi 300

# Use different model
python pdf_to_markdown.py document.pdf --model microsoft/Phi-3-vision-128k-instruct

# Disable quantization (requires more GPU memory)
python pdf_to_markdown.py document.pdf --no-quantization
```

### Command-Line Arguments

- `pdf_path`: Path to the PDF file (required)
- `--output-dir`: Output directory for results (default: `results/`)
- `--dpi`: DPI for image extraction (default: 200)
- `--model`: Hugging Face model to use (default: `microsoft/Phi-3.5-vision-instruct`)
- `--no-quantization`: Disable 4-bit quantization

## Output

The script creates:

1. **Individual page files**: `{pdf_name}_page_001.md`, `{pdf_name}_page_002.md`, etc.
2. **Combined document**: `{pdf_name}_complete.md` with all pages

Each Markdown file includes:
- Section and subsection headers (H1, H2, H3, etc.)
- Tables in Markdown format
- Lists (bullet points and numbered)
- Code blocks
- Text formatting (bold, italic)

## Model Information

**Phi-3.5 Vision Instruct** (quantized to 4-bit):
- Parameters: ~4.2B (vision + language)
- Quantized size: ~2-3GB
- Context length: 128K tokens
- Strong at document understanding and structure detection

## Performance

- **GPU (CUDA)**: ~10-30 seconds per page (depending on complexity)
- **CPU**: ~1-3 minutes per page
- **Memory**: 4-8GB with quantization, 12-16GB without

## Troubleshooting

### CUDA Out of Memory
- Reduce DPI: `--dpi 150`
- Ensure quantization is enabled (default)
- Process fewer pages at a time

### Poor OCR Quality
- Increase DPI: `--dpi 300`
- Ensure PDF has good quality scans
- Try different models

### Slow Processing
- Ensure CUDA is available and working
- Reduce DPI for faster processing
- Use quantization (enabled by default)

## Example

```bash
# Convert a research paper
python pdf_to_markdown.py research_paper.pdf --dpi 250 --output-dir papers/

# Convert a book with high quality
python pdf_to_markdown.py book.pdf --dpi 300 --output-dir books/
```

## License

MIT License

## Credits

- Model: Microsoft Phi-3.5 Vision
- PDF Processing: pdf2image, Pillow
- Quantization: bitsandbytes, Hugging Face Transformers