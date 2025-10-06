#!/usr/bin/env python3
"""
Example usage of the PDF to Markdown converter
"""

from pdf_to_markdown import PDFToMarkdownConverter
from pathlib import Path


def convert_single_pdf():
    """Example: Convert a single PDF file"""
    
    # Initialize converter with quantization
    converter = PDFToMarkdownConverter(
        model_name="microsoft/Phi-3.5-vision-instruct",
        use_quantization=True
    )
    
    # Convert PDF to Markdown
    pdf_file = "example.pdf"  # Replace with your PDF path
    
    if Path(pdf_file).exists():
        markdown_pages = converter.convert_pdf_to_markdown(
            pdf_path=pdf_file,
            output_dir="results",
            dpi=200
        )
        
        print(f"\nConverted {len(markdown_pages)} pages successfully!")
    else:
        print(f"PDF file not found: {pdf_file}")
        print("Please provide a valid PDF file path.")


def convert_multiple_pdfs():
    """Example: Convert multiple PDF files"""
    
    # Initialize converter once
    converter = PDFToMarkdownConverter(
        model_name="microsoft/Phi-3.5-vision-instruct",
        use_quantization=True
    )
    
    # List of PDFs to convert
    pdf_files = [
        "document1.pdf",
        "document2.pdf",
        "document3.pdf"
    ]
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            print(f"\n{'='*60}")
            print(f"Converting: {pdf_file}")
            print(f"{'='*60}\n")
            
            converter.convert_pdf_to_markdown(
                pdf_path=pdf_file,
                output_dir=f"results/{Path(pdf_file).stem}",
                dpi=200
            )
        else:
            print(f"Skipping {pdf_file} - file not found")


if __name__ == "__main__":
    print("PDF to Markdown Converter - Example Usage\n")
    
    # Choose which example to run
    print("1. Convert single PDF")
    print("2. Convert multiple PDFs")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        convert_single_pdf()
    elif choice == "2":
        convert_multiple_pdfs()
    else:
        print("Invalid choice. Running single PDF example...")
        convert_single_pdf()