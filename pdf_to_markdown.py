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


class PDFToMarkdownConverter:
    def __init__(self, model_name="microsoft/Phi-3.5-vision-instruct", use_quantization=True):
        """
        Initialize the PDF to Markdown converter.
        
        Args:
            model_name: Hugging Face model identifier
            use_quantization: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model with 8-bit quantization
        print(f"Loading model: {model_name}")
        if use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                _attn_implementation='eager'
            )
            print("Model loaded with 8-bit quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print("Model loaded without quantization")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def extract_pages_as_images(self, pdf_path, dpi=200):
        """
        Extract all pages from PDF as images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image extraction
            
        Returns:
            List of PIL Image objects
        """
        print(f"Extracting pages from PDF: {pdf_path}")
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            print(f"Extracted {len(images)} pages")
            return images
        except Exception as e:
            print(f"Error extracting pages: {e}")
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
            # Process image and prompt
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Remove input tokens from generated output
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            
            # Decode the response
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
        
        except Exception as e:
            print(f"Error processing page {page_number}: {e}")
            return f"# Error Processing Page {page_number}\n\nFailed to extract content from this page."
    
    def convert_pdf_to_markdown(self, pdf_path, output_dir="results", dpi=200):
        """
        Convert entire PDF to Markdown files.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for Markdown files
            dpi: Resolution for image extraction
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get PDF name for output files
        pdf_name = Path(pdf_path).stem
        
        # Extract pages as images
        pages = self.extract_pages_as_images(pdf_path, dpi=dpi)
        
        # Process each page
        print("\nConverting pages to Markdown...")
        all_markdown = []
        
        for i, page_image in enumerate(tqdm(pages, desc="Processing pages")):
            page_num = i + 1
            
            # Convert to Markdown
            markdown_content = self.image_to_markdown(page_image, page_num)
            all_markdown.append(markdown_content)
            
            # Save individual page
            page_output_file = output_path / f"{pdf_name}_page_{page_num:03d}.md"
            with open(page_output_file, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Page {page_num} -->\n\n")
                f.write(markdown_content)
                f.write("\n")
            
            print(f"Saved: {page_output_file}")
        
        # Save combined document
        combined_output_file = output_path / f"{pdf_name}_complete.md"
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            for i, content in enumerate(all_markdown, 1):
                f.write(f"<!-- Page {i} -->\n\n")
                f.write(content)
                f.write("\n\n---\n\n")
        
        print(f"\nConversion complete!")
        print(f"Individual pages saved to: {output_path}")
        print(f"Combined document saved to: {combined_output_file}")
        
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
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return
    
    # Initialize converter
    converter = PDFToMarkdownConverter(
        model_name=args.model,
        use_quantization=not args.no_quantization
    )
    
    # Convert PDF
    converter.convert_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()