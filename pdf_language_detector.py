import os
import re
import argparse
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import fitz  # PyMuPDF
from lingua import Language, LanguageDetectorBuilder

@dataclass
class LanguageBlock:
    """Represents a block of text in a specific language with its position on the page"""
    language: Language
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
    confidence: float

class PDFLanguageDetector:
    def __init__(self, languages=None, min_confidence=0.65, min_text_length=20):
        """
        Initialize the PDF language detector
        
        Args:
            languages: List of Language objects to detect. If None, uses all spoken languages.
            min_confidence: Minimum confidence threshold for language detection
            min_text_length: Minimum text length to attempt language detection
        """
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        
        # Initialize language detector
        if languages:
            self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        else:
            # Use all spoken languages for better accuracy
            self.detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    
    def process_pdf(self, pdf_path: str) -> List[LanguageBlock]:
        """
        Process a PDF file and detect languages with their bounding boxes
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of LanguageBlock objects containing detected languages and their positions
        """
        language_blocks = []
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Extract text blocks with their bounding boxes
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # block structure is (x0, y0, x1, y1, text, block_no, block_type)
                text = block[4]
                bbox = block[:4]  # x0, y0, x1, y1
                
                # Skip if text is too short
                if len(text.strip()) < self.min_text_length:
                    continue
                
                # Detect language
                detected_language = self._detect_language(text)
                if detected_language:
                    language, confidence = detected_language
                    language_blocks.append(
                        LanguageBlock(
                            language=language,
                            text=text,
                            bbox=bbox,
                            page_num=page_num,
                            confidence=confidence
                        )
                    )
            
            # Also try to detect multiple languages within each block
            # This is helpful for mixed-language documents
            for block in blocks:
                text = block[4]
                bbox = block[:4]
                
                # Skip if text is too short
                if len(text.strip()) < self.min_text_length * 2:  # Require longer text for multi-detection
                    continue
                
                # Detect multiple languages
                multi_results = self.detector.detect_multiple_languages_of(text)
                
                for result in multi_results:
                    sub_text = text[result.start_index:result.end_index]
                    # Skip if sub-text is too short
                    if len(sub_text.strip()) < self.min_text_length:
                        continue
                    
                    # Calculate approximate bounding box for this text segment
                    # This is an approximation as we don't have exact character positions
                    relative_start = result.start_index / len(text)
                    relative_end = result.end_index / len(text)
                    sub_bbox = (
                        bbox[0] + (bbox[2] - bbox[0]) * relative_start,
                        bbox[1],
                        bbox[0] + (bbox[2] - bbox[0]) * relative_end,
                        bbox[3]
                    )
                    
                    # Compute confidence (currently not provided directly by detect_multiple_languages_of)
                    confidence = self.detector.compute_language_confidence(sub_text, result.language)
                    
                    if confidence >= self.min_confidence:
                        language_blocks.append(
                            LanguageBlock(
                                language=result.language,
                                text=sub_text,
                                bbox=sub_bbox,
                                page_num=page_num,
                                confidence=confidence
                            )
                        )
        
        return language_blocks
    
    def _detect_language(self, text: str) -> Optional[Tuple[Language, float]]:
        """
        Detect the language of a text snippet
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (Language, confidence) or None if confidence is below threshold
        """
        # Get most likely language
        language = self.detector.detect_language_of(text)
        if not language:
            return None
        
        # Get confidence value
        confidence = self.detector.compute_language_confidence(text, language)
        
        if confidence >= self.min_confidence:
            return language, confidence
        return None
    
    def create_annotated_pdf(self, pdf_path: str, output_path: str, language_blocks: List[LanguageBlock]) -> None:
        """
        Create a new PDF with language annotations
        
        Args:
            pdf_path: Source PDF path
            output_path: Output PDF path
            language_blocks: List of detected language blocks
        """
        # Open source PDF
        doc = fitz.open(pdf_path)
        
        # Create colors for different languages
        colors = self._generate_colors(language_blocks)
        
        # Group blocks by page
        blocks_by_page = {}
        for block in language_blocks:
            if block.page_num not in blocks_by_page:
                blocks_by_page[block.page_num] = []
            blocks_by_page[block.page_num].append(block)
        
        # Add annotations to each page
        for page_num, page in enumerate(doc):
            if page_num not in blocks_by_page:
                continue
                
            blocks = blocks_by_page[page_num]
            
            for block in blocks:
                # Create rectangle for the block
                rect = fitz.Rect(block.bbox)
                
                # Add colored rectangle with increased transparency
                color = colors.get(block.language, (1, 0, 0))  # Default to red
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color, fill=color + (0.1,))  # Lower alpha value (0.1) for more transparency
                annot.set_opacity(0.3)  # Set overall opacity to make both stroke and fill more transparent
                annot.update()
                
                # Add text annotation with language info
                point = fitz.Point(rect.x0, rect.y0)
                info = f"{block.language.name} ({block.confidence:.2f})"
                text_annot = page.add_text_annot(point, info)
                text_annot.update()
        
        # Save the annotated PDF
        doc.save(output_path)
        doc.close()
    
    def _generate_colors(self, language_blocks: List[LanguageBlock]) -> Dict[Language, Tuple[float, float, float]]:
        """
        Generate distinct colors for each language
        
        Args:
            language_blocks: List of language blocks
        
        Returns:
            Dictionary mapping languages to RGB color tuples
        """
        import colorsys
        
        # Get unique languages
        languages = set(block.language for block in language_blocks)
        
        # Generate evenly spaced colors
        colors = {}
        for i, language in enumerate(languages):
            hue = i / len(languages)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colors[language] = (r, g, b)
        
        return colors


def main():
    parser = argparse.ArgumentParser(description="Detect and annotate languages in PDF documents")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output PDF path (default: adds '_annotated' to filename)")
    parser.add_argument("--languages", "-l", nargs="+", help="List of language codes to detect (e.g., EN FR DE)")
    parser.add_argument("--confidence", "-c", type=float, default=0.65, help="Minimum confidence threshold (0-1)")
    parser.add_argument("--min-length", "-m", type=int, default=20, help="Minimum text length for detection")
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        pdf_name, pdf_ext = os.path.splitext(args.pdf_path)
        args.output = f"{pdf_name}_annotated{pdf_ext}"
    
    # Convert language codes to Language objects if specified
    languages = None
    if args.languages:
        languages = []
        for lang_code in args.languages:
            try:
                # Try as ISO 639-1 code first
                language = Language.from_iso_code_639_1(lang_code.upper())
                languages.append(language)
            except ValueError:
                try:
                    # Try as language name
                    language = Language.from_str(lang_code.upper())
                    languages.append(language)
                except ValueError:
                    print(f"Warning: Unrecognized language code or name: {lang_code}")
    
    # Initialize detector
    detector = PDFLanguageDetector(
        languages=languages,
        min_confidence=args.confidence,
        min_text_length=args.min_length
    )
    
    # Process PDF
    print(f"Processing {args.pdf_path}...")
    language_blocks = detector.process_pdf(args.pdf_path)
    
    # Display results
    print(f"Found {len(language_blocks)} language blocks:")
    for i, block in enumerate(language_blocks[:10]):  # Show first 10 blocks
        print(f"{i+1}. {block.language.name} (confidence: {block.confidence:.2f})")
        print(f"   Text: {block.text[:50]}..." if len(block.text) > 50 else f"   Text: {block.text}")
    
    if len(language_blocks) > 10:
        print(f"... and {len(language_blocks) - 10} more")
    
    # Create annotated PDF
    print(f"Creating annotated PDF: {args.output}")
    detector.create_annotated_pdf(args.pdf_path, args.output, language_blocks)
    print("Done!")


if __name__ == "__main__":
    main()
