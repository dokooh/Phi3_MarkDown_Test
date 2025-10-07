#!/usr/bin/env python3
"""
PDF to Markdown Converter using Phi-3 14B
Extracts pages from PDF as images and converts them to Markdown using Hugging Face's Phi-3 model.
"""

import os
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm


def parse_page_ranges(page_string):
    """
    Parse page range string into a list of page numbers.
    
    Args:
        page_string: String like '1-3,5,7-10' or 'all' or None
        
    Returns:
        List of page numbers (1-indexed) or None for all pages
    """
    if not page_string or page_string.lower() == 'all':
        print("DEBUG: Page range set to 'all' - will process all pages")
        return None
    
    pages = []
    print(f"DEBUG: Parsing page range string: '{page_string}'")
    
    try:
        # Split by comma and process each part
        parts = page_string.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Range like '1-3'
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                range_pages = list(range(start, end + 1))
                pages.extend(range_pages)
                print(f"DEBUG: Added page range {start}-{end}: {range_pages}")
            else:
                # Single page
                page = int(part)
                pages.append(page)
                print(f"DEBUG: Added single page: {page}")
        
        # Remove duplicates and sort
        pages = sorted(list(set(pages)))
        print(f"DEBUG: Final page list (sorted, deduplicated): {pages}")
        return pages
    
    except Exception as e:
        print(f"ERROR: Invalid page range format '{page_string}': {e}")
        print("Expected format: '1-3,5,7-10' or 'all'")
        raise ValueError(f"Invalid page range format: {page_string}")


class PDFToMarkdownConverter:
    def __init__(self, model_name="microsoft/Phi-3.5-vision-instruct", use_quantization=True):
        """
        Initialize the PDF to Markdown converter.
        
        Args:
            model_name: Hugging Face model identifier
            use_quantization: Whether to use 8-bit quantization
        """
        print("DEBUG: Initializing PDFToMarkdownConverter...")
        print(f"DEBUG: Model name: {model_name}")
        print(f"DEBUG: Use quantization: {use_quantization}")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEBUG: CUDA available: {torch.cuda.is_available()}")
        print(f"DEBUG: Using device: {self.device}")
        
        # Load model with 8-bit quantization
        print(f"DEBUG: Loading model: {model_name}")
        if use_quantization and self.device == "cuda":
            print("DEBUG: Setting up 8-bit quantization config...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            print("DEBUG: Loading model with quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                _attn_implementation='eager'
            )
            print("DEBUG: Model loaded with 8-bit quantization")
        else:
            print("DEBUG: Loading model without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print("DEBUG: Model loaded without quantization")
        
        # Load processor
        print("DEBUG: Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("DEBUG: Model and processor loaded successfully!")
        print("DEBUG: Initialization complete.")
    
    def extract_pages_as_images(self, pdf_path, dpi=200, page_numbers=None):
        """
        Extract specified pages from PDF as images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image extraction
            page_numbers: List of page numbers to extract (1-indexed), or None for all
            
        Returns:
            List of PIL Image objects and list of actual page numbers
        """
        print(f"DEBUG: Starting page extraction from PDF: {pdf_path}")
        print(f"DEBUG: DPI setting: {dpi}")
        print(f"DEBUG: Requested page numbers: {page_numbers}")
        
        try:
            print("DEBUG: Converting PDF to images...")
            all_images = convert_from_path(pdf_path, dpi=dpi)
            total_pages = len(all_images)
            print(f"DEBUG: PDF contains {total_pages} total pages")
            
            if page_numbers is None:
                print("DEBUG: No specific pages requested, returning all pages")
                actual_page_numbers = list(range(1, total_pages + 1))
                return all_images, actual_page_numbers
            
            # Filter specific pages
            print("DEBUG: Filtering specific pages...")
            selected_images = []
            actual_page_numbers = []
            
            for page_num in page_numbers:
                if 1 <= page_num <= total_pages:
                    # Convert to 0-indexed for list access
                    image_index = page_num - 1
                    selected_images.append(all_images[image_index])
                    actual_page_numbers.append(page_num)
                    print(f"DEBUG: Selected page {page_num}")
                else:
                    print(f"WARNING: Page {page_num} is out of range (PDF has {total_pages} pages), skipping")
            
            print(f"DEBUG: Successfully extracted {len(selected_images)} pages: {actual_page_numbers}")
            return selected_images, actual_page_numbers
            
        except Exception as e:
            print(f"ERROR: Failed to extract pages: {e}")
            raise
    
    def image_to_markdown(self, image, page_number):
        """
        Convert a single page image to Markdown using Phi-3 Vision.
        
        Args:
            image: PIL Image object
            page_number: Page number for context
            
        Returns:
            Markdown string
        """
        print(f"DEBUG: Starting conversion of page {page_number} to Markdown")
        print(f"DEBUG: Image size: {image.size}")
        
        # Prepare the prompt
        prompt = f"""<|user|>
<|image_1|>
You are an expert at analyzing document images and converting them to Markdown format.

Please analyze this document page (page {page_number}) and convert its content to well-structured Markdown.

Instructions:
1. Detect all sections and subsections with appropriate heading levels (# for main sections, ## for subsections, etc.)
2. Preserve the document hierarchy and structure
3. Convert tables to Markdown table format if present
4. Include bullet points and numbered lists as they appear
5. Preserve emphasis (bold, italic) where visible
6. Include code blocks if any code is present
7. Maintain proper spacing and formatting

Output only the Markdown content without any explanations or preamble.<|end|>
<|assistant|>
"""
        
        try:
            print(f"DEBUG: Processing image and prompt for page {page_number}...")
            # Process image and prompt
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
            print(f"DEBUG: Input tensor shape: {inputs['input_ids'].shape}")
            
            print(f"DEBUG: Starting model generation for page {page_number}...")
            # Generate output
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to avoid DynamicCache issues
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    past_key_values=None,  # Explicitly set past key values to None
                )
            
            print(f"DEBUG: Generation complete for page {page_number}, decoding response...")
            # Remove input tokens from generated output
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            
            # Decode the response
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            response_text = response.strip()
            print(f"DEBUG: Successfully converted page {page_number} to Markdown ({len(response_text)} characters)")
            return response_text
        
        except Exception as e:
            print(f"ERROR: Failed to process page {page_number}: {e}")
            return f"# Error Processing Page {page_number}\n\nFailed to extract content from this page."
    
    def convert_pdf_to_markdown(self, pdf_path, output_dir="results", dpi=200, page_numbers=None):
        """
        Convert PDF to Markdown files.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for Markdown files
            dpi: Resolution for image extraction
            page_numbers: List of page numbers to extract (1-indexed), or None for all
        """
        print(f"DEBUG: Starting PDF conversion process...")
        print(f"DEBUG: Input PDF: {pdf_path}")
        print(f"DEBUG: Output directory: {output_dir}")
        print(f"DEBUG: DPI: {dpi}")
        print(f"DEBUG: Page numbers: {page_numbers}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: Created output directory: {output_path}")
        
        # Get PDF name for output files
        pdf_name = Path(pdf_path).stem
        print(f"DEBUG: PDF base name: {pdf_name}")
        
        # Extract pages as images
        print("DEBUG: Calling extract_pages_as_images...")
        pages, actual_page_numbers = self.extract_pages_as_images(pdf_path, dpi=dpi, page_numbers=page_numbers)
        print(f"DEBUG: Extracted {len(pages)} pages: {actual_page_numbers}")
        
        # Process each page
        print("DEBUG: Starting page-by-page Markdown conversion...")
        all_markdown = []
        
        for i, (page_image, actual_page_num) in enumerate(zip(pages, actual_page_numbers)):
            print(f"DEBUG: Processing page {actual_page_num} ({i+1}/{len(pages)})")
            
            # Convert to Markdown
            markdown_content = self.image_to_markdown(page_image, actual_page_num)
            all_markdown.append(markdown_content)
            
            # Save individual page
            page_output_file = output_path / f"{pdf_name}_page_{actual_page_num:03d}.md"
            print(f"DEBUG: Saving individual page to: {page_output_file}")
            with open(page_output_file, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Page {actual_page_num} -->\n\n")
                f.write(markdown_content)
                f.write("\n")
            
            print(f"DEBUG: Successfully saved page {actual_page_num}")
        
        # Save combined document
        combined_output_file = output_path / f"{pdf_name}_complete.md"
        print(f"DEBUG: Saving combined document to: {combined_output_file}")
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            for content, actual_page_num in zip(all_markdown, actual_page_numbers):
                f.write(f"<!-- Page {actual_page_num} -->\n\n")
                f.write(content)
                f.write("\n\n---\n\n")
        
        print(f"DEBUG: Conversion process completed successfully!")
        print(f"DEBUG: Processed {len(pages)} pages")
        print(f"DEBUG: Individual pages saved to: {output_path}")
        print(f"DEBUG: Combined document saved to: {combined_output_file}")
        
        return all_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to Markdown using Phi-3 Vision model with 8-bit quantization"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to convert"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for Markdown files (default: results/)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3.5-vision-instruct",
        help="Hugging Face model to use (default: microsoft/Phi-3.5-vision-instruct)"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 8-bit quantization (uses more memory but may improve quality)"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Specific pages to extract (e.g., '1-3,5,7-10' or 'all'). Default: all pages"
    )
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return
    
    # Parse page ranges
    page_numbers = parse_page_ranges(args.pages)
    
    # Initialize converter
    converter = PDFToMarkdownConverter(
        model_name=args.model,
        use_quantization=not args.no_quantization
    )
    
    # Convert PDF
    converter.convert_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        dpi=args.dpi,
        page_numbers=page_numbers
    )


if __name__ == "__main__":
    main()
