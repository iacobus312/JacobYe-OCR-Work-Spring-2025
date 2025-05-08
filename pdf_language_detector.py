import os
import re
import argparse
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict

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
    block_num: int  # Tracking the original block number for ordering

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
            
            # Print number of blocks per page for first few pages (debugging)
            if page_num < 5:
                print(f"DEBUG: Page {page_num+1} has {len(blocks)} text blocks")
            
            for block_idx, block in enumerate(blocks):
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
                            confidence=confidence,
                            block_num=block_idx
                        )
                    )
                
            # Also try working with sentences instead of blocks for better detection
            # This helps in cases where the PDF's block structure doesn't match language boundaries
            sentences = self._extract_sentences(page.get_text())
            for i, sentence in enumerate(sentences):
                # Skip short sentences
                if len(sentence.strip()) < self.min_text_length:
                    continue
                
                # Use simple estimation for sentence position (this is approximate)
                text_instances = page.search_for(sentence[:min(20, len(sentence))])
                if text_instances:
                    bbox = text_instances[0]  # Use first match position
                else:
                    # If text not found (might happen due to spacing, etc.), use approximate position
                    bbox = (50, 50 + (i * 20), 500, 70 + (i * 20))
                
                # Detect language
                detected_language = self._detect_language(sentence)
                if detected_language:
                    language, confidence = detected_language
                    
                    # Add sentence as a separate language block
                    language_blocks.append(
                        LanguageBlock(
                            language=language,
                            text=sentence,
                            bbox=bbox,
                            page_num=page_num,
                            confidence=confidence,
                            block_num=1000 + i  # Use high numbers to differentiate from block-based detection
                        )
                    )
        
        return language_blocks
        
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, trying to respect language boundaries
        
        Args:
            text: The text to split into sentences
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting using common sentence terminators
        # This is a simplified approach - for production, consider using a NLP library
        sentences = []
        
        # Handle common sentence-ending punctuation in various languages
        # Include period, question mark, exclamation mark, and their Asian equivalents
        terminators = ['.', '!', '?', '。', '！', '？', '\n\n']
        
        # Split by terminators
        current_sentence = ""
        for char in text:
            current_sentence += char
            
            # Check if we reached a sentence end
            if char in terminators:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add the last sentence if there's content
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences
    
    def _detect_language(self, text: str) -> Optional[Tuple[Language, float]]:
        """
        Detect the language of a text snippet
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (Language, confidence) or None if confidence is below threshold
        """
        # Clean the text to improve detection
        cleaned_text = self._clean_text_for_detection(text)
        
        # Skip if text is too short after cleaning
        if len(cleaned_text) < self.min_text_length:
            return None
        
        # Get most likely language
        try:
            language = self.detector.detect_language_of(cleaned_text)
            if not language:
                return None
            
            # Get confidence value
            confidence = self.detector.compute_language_confidence(cleaned_text, language)
            
            if confidence >= self.min_confidence:
                return language, confidence
        except Exception as e:
            print(f"Error detecting language: {e}")
            return None
            
        return None
        
    def _clean_text_for_detection(self, text: str) -> str:
        """
        Clean and prepare text for language detection
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common non-language-specific elements that can confuse detection
        # Like page numbers, URLs, etc.
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'\d+[.\-]\d+', '', text)     # Remove numbers with dots/dashes
        text = re.sub(r'\b\d+\b', '', text)         # Remove standalone numbers
        
        # Remove punctuation marks that could confuse language detection
        # but preserve
    
    def find_alternating_language_pages(self, language_blocks: List[LanguageBlock], 
                                       min_alternations: int = 2) -> Dict[int, List[Tuple[Language, Language]]]:
        """
        Find pages with alternating languages
        
        Args:
            language_blocks: List of detected language blocks
            min_alternations: Minimum number of language alternations needed to consider a page
            
        Returns:
            Dictionary mapping page numbers to lists of language alternation pairs
        """
        # Group blocks by page
        blocks_by_page = defaultdict(list)
        for block in language_blocks:
            blocks_by_page[block.page_num].append(block)
        
        # Sort blocks within each page by their natural reading order (usually top-to-bottom)
        for page_num in blocks_by_page:
            blocks_by_page[page_num].sort(key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by y0, then x0
        
        # Find pages with alternating languages
        alternating_pages = {}
        
        for page_num, blocks in blocks_by_page.items():
            alternations = []
            
            if len(blocks) < 2:
                continue
                
            # Check for alternations by comparing consecutive blocks
            for i in range(len(blocks) - 1):
                current_lang = blocks[i].language
                next_lang = blocks[i+1].language
                
                # If languages are different, record the alternation
                if current_lang != next_lang:
                    alternations.append((current_lang, next_lang))
            
            # If there are enough alternations, record this page
            if len(alternations) >= min_alternations:
                alternating_pages[page_num] = alternations
        
        return alternating_pages
    
    def find_repeated_alternation_pages(self, language_blocks: List[LanguageBlock], 
                                       languages: Set[Language] = None, 
                                       min_repeat_count: int = 2) -> Dict[int, List[Language]]:
        """
        Find pages where specific languages alternate repeatedly (e.g., Chinese→English→Chinese→English)
        
        Args:
            language_blocks: List of detected language blocks
            languages: Set of languages to look for alternations between (if None, detect automatically)
            min_repeat_count: Minimum number of alternations needed (not complete cycles)
            
        Returns:
            Dictionary mapping page numbers to lists of alternating languages
        """
        # Group blocks by page
        blocks_by_page = defaultdict(list)
        for block in language_blocks:
            blocks_by_page[block.page_num].append(block)
        
        # Sort blocks within each page by their natural reading order
        for page_num in blocks_by_page:
            blocks_by_page[page_num].sort(key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by y0, then x0
        
        # Find pages with repeated alternating language patterns
        repeating_pages = {}
        
        for page_num, blocks in blocks_by_page.items():
            if len(blocks) < 2:  # Need at least 2 blocks for language alternation
                continue
            
            # Extract language sequence
            lang_sequence = [block.language for block in blocks]
            
            # If specific languages not provided, use all languages on the page
            if not languages:
                page_languages = set(lang_sequence)
            else:
                page_languages = languages
                
            # For clearer debugging
            if page_num < 10:  # Only print first few pages to avoid clutter
                languages_on_page = ", ".join([lang.name for lang in set(lang_sequence)])
                print(f"DEBUG: Page {page_num+1} has languages: {languages_on_page}")
                
            # Only consider pages with at least 2 languages
            if len(set(lang_sequence)) < 2:
                continue
                
            # Count alternations between any languages
            alternation_count = 0
            for i in range(len(lang_sequence) - 1):
                current_lang = lang_sequence[i]
                next_lang = lang_sequence[i+1]
                
                # If languages are different, count as alternation
                if current_lang != next_lang:
                    alternation_count += 1
                    
            if page_num < 10:  # Debugging
                print(f"DEBUG: Page {page_num+1} has {alternation_count} language alternations")
                
            # Consider the page as having alternating languages if it has enough alternations
            # This is a more lenient approach than requiring perfect A→B→A→B patterns
            if alternation_count >= min_repeat_count:
                # Find the two most common languages for reporting
                lang_counts = {}
                for lang in lang_sequence:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                # Take the two (or more) most common languages
                sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
                main_languages = [lang for lang, count in sorted_langs[:min(3, len(sorted_langs))]]
                
                repeating_pages[page_num] = main_languages
        
        return repeating_pages
    
    def create_annotated_pdf(self, pdf_path: str, output_path: str, language_blocks: List[LanguageBlock],
                          alternating_pages: Dict[int, List[Language]] = None) -> None:
        """
        Create a new PDF with language annotations, highlighting pages with alternating languages
        
        Args:
            pdf_path: Source PDF path
            output_path: Output PDF path
            language_blocks: List of detected language blocks
            alternating_pages: Dictionary of pages with alternating languages
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
            # Highlight alternating pages with a border
            if alternating_pages and page_num in alternating_pages:
                # Add a prominent border to highlight the page
                rect = page.rect
                border_annot = page.add_rect_annot(rect)
                border_annot.set_colors(stroke=(1, 0, 0))  # Red border
                border_annot.set_border(width=3)  # Thicker border
                border_annot.update()
                
                # Add a note explaining the alternation
                langs = alternating_pages[page_num]
                lang_names = [lang.name for lang in langs]
                note_text = f"ALTERNATING LANGUAGES: {' ↔ '.join(lang_names)}"
                text_point = fitz.Point(rect.x0 + 10, rect.y0 + 10)
                note_annot = page.add_text_annot(text_point, note_text)
                note_annot.set_colors(stroke=(1, 0, 0))  # Red text
                note_annot.update()
            
            # Add language block annotations
            if page_num in blocks_by_page:
                blocks = blocks_by_page[page_num]
                
                for block in blocks:
                    # Create rectangle for the block
                    rect = fitz.Rect(block.bbox)
                    
                    # Add colored rectangle with increased transparency
                    color = colors.get(block.language, (1, 0, 0))  # Default to red
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=color, fill=color + (0.1,))  # Lower alpha for transparency
                    annot.set_opacity(0.3)
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
            hue = i / max(1, len(languages))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colors[language] = (r, g, b)
        
        return colors


def main():
    parser = argparse.ArgumentParser(description="Detect and annotate alternating languages in PDF documents")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output PDF path (default: adds '_annotated' to filename)")
    parser.add_argument("--report", "-r", help="Output path for the alternating pages report (default: adds '_report.txt' to filename)")
    parser.add_argument("--languages", "-l", nargs="+", help="List of language codes to detect (e.g., EN ZH)")
    parser.add_argument("--confidence", "-c", type=float, default=0.55, help="Minimum confidence threshold (0-1)")
    parser.add_argument("--min-length", "-m", type=int, default=10, help="Minimum text length for detection")
    parser.add_argument("--min-alternations", "-a", type=int, default=1, 
                       help="Minimum number of language alternations to detect (default: 1)")
    parser.add_argument("--annotate", action="store_true", help="Create an annotated PDF with highlights")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--page-range", help="Process only specific pages (format: start-end)")
    
    args = parser.parse_args()
    
    # Set default output paths if not specified
    if not args.output and args.annotate:
        pdf_name, pdf_ext = os.path.splitext(args.pdf_path)
        args.output = f"{pdf_name}_annotated{pdf_ext}"
    
    if not args.report:
        pdf_name, _ = os.path.splitext(args.pdf_path)
        args.report = f"{pdf_name}_alternating_pages_report.txt"
    
    # Parse page range if specified
    page_range = None
    if args.page_range:
        try:
            parts = args.page_range.split('-')
            if len(parts) == 2:
                start_page = int(parts[0]) - 1  # Convert from 1-based to 0-based
                end_page = int(parts[1]) - 1
                page_range = (start_page, end_page)
            else:
                print(f"Warning: Invalid page range format '{args.page_range}'. Processing all pages.")
        except ValueError:
            print(f"Warning: Invalid page range '{args.page_range}'. Processing all pages.")
    
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
    
    # Get PDF information first
    doc = fitz.open(args.pdf_path)
    total_pages = len(doc)
    doc.close()
    
    print(f"PDF has {total_pages} pages.")
    
    # Process the PDF to detect languages
    language_blocks = detector.process_pdf(args.pdf_path)
    
    # Count unique languages detected
    all_languages = set(block.language for block in language_blocks)
    lang_names = [lang.name for lang in all_languages]
    print(f"Detected {len(all_languages)} languages: {', '.join(lang_names)}")
    
    # Find pages with alternating languages
    print("Finding pages with alternating languages...")
    
    # Use more lenient detection settings to address the underperformance
    alternating_pages = detector.find_repeated_alternation_pages(
        language_blocks, 
        languages=set(languages) if languages else None,
        min_repeat_count=args.min_alternations
    )
    
    # Also try the simpler alternation detection as a fallback
    if len(alternating_pages) < total_pages * 0.05:  # If we found fewer than 5% of pages
        print("Initial detection found few pages. Trying simpler alternation detection...")
        simple_alternating = detector.find_alternating_language_pages(
            language_blocks,
            min_alternations=args.min_alternations
        )
        
        # Merge the results
        for page_num, alternations in simple_alternating.items():
            if page_num not in alternating_pages and alternations:
                # Extract the unique languages involved in alternations
                langs = set()
                for lang_pair in alternations:
                    langs.add(lang_pair[0])
                    langs.add(lang_pair[1])
                alternating_pages[page_num] = list(langs)
    
    # Display results
    print(f"Found {len(alternating_pages)} pages with alternating languages:")
    if alternating_pages:
        # Sort page numbers for cleaner output
        sorted_pages = sorted(alternating_pages.keys())
        
        # Print the first 10 pages and the count of remaining pages
        for page_num in sorted_pages[:10]:
            langs = alternating_pages[page_num]
            lang_names = [lang.name for lang in langs]
            print(f"Page {page_num + 1}: Alternating between {' and '.join(lang_names)}")
            
        if len(sorted_pages) > 10:
            print(f"... and {len(sorted_pages) - 10} more pages")
    else:
        print("No pages with significant language alternation patterns found.")
    
    # Write detailed report file
    with open(args.report, 'w') as f:
        f.write(f"Language Alternation Report for {args.pdf_path}\n")
        f.write(f"=================================================\n\n")
        
        # First, write summary information
        f.write("Summary:\n")
        f.write(f"- Total pages in document: {total_pages}\n")
        f.write(f"- Pages with alternating languages: {len(alternating_pages)}\n")
        f.write(f"- Languages detected: {', '.join(lang_names)}\n\n")
        
        if alternating_pages:
            f.write(f"Pages with alternating languages:\n\n")
            sorted_pages = sorted(alternating_pages.keys())
            
            # Group by consecutive page ranges for cleaner output
            page_ranges = []
            current_range_start = sorted_pages[0]
            prev_page = sorted_pages[0]
            
            for page in sorted_pages[1:]:
                if page > prev_page + 1:  # Gap in sequence
                    page_ranges.append((current_range_start, prev_page))
                    current_range_start = page
                prev_page = page
                
            # Add the last range
            page_ranges.append((current_range_start, prev_page))
            
            # Write out the ranges
            f.write("Page ranges with alternating languages:\n")
            for start, end in page_ranges:
                if start == end:
                    f.write(f"- Page {start + 1}\n")
                else:
                    f.write(f"- Pages {start + 1}-{end + 1}\n")
            f.write("\n")
            
            # Write detailed page-by-page information
            f.write("Detailed page information:\n\n")
            for page_num in sorted_pages:
                langs = alternating_pages[page_num]
                lang_names = [lang.name for lang in langs]
                f.write(f"Page {page_num + 1}: Alternating between {' and '.join(lang_names)}\n")
                
                # Count blocks for each language on this page
                page_blocks = [b for b in language_blocks if b.page_num == page_num]
                lang_counts = {}
                for block in page_blocks:
                    if block.language in langs:
                        lang_counts[block.language] = lang_counts.get(block.language, 0) + 1
                
                # Add block counts to the report
                f.write("  Language distribution:\n")
                for lang, count in lang_counts.items():
                    f.write(f"  - {lang.name}: {count} blocks\n")
                
                f.write("\n")
        else:
            f.write("No pages with significant language alternation patterns found.\n")
            f.write("\nTroubleshooting suggestions:\n")
            f.write("1. Try lowering the confidence threshold (--confidence)\n")
            f.write("2. Try reducing the minimum text length (--min-length)\n")
            f.write("3. Specify the exact languages to look for (--languages)\n")
    
    print(f"Report written to {args.report}")
    
    # Create annotated PDF if requested
    if args.annotate:
        print(f"Creating annotated PDF: {args.output}")
        detector.create_annotated_pdf(args.pdf_path, args.output, language_blocks, alternating_pages)
    
    print("Done!")


if __name__ == "__main__":
    main()