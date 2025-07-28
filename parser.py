import fitz 
import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

class PDFHeadingParser:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        self.text_blocks = []
        self.heading_patterns = [
            r'^\d+\.?\s+[A-Z]', 
            r'^[A-Z][A-Z\s]{2,}$',  
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$', 
            r'^\d+\.\d+\.?\s+',  
            r'^[IVX]+\.?\s+[A-Z]', 
            r'^[A-Z]\.\s+[A-Z]',
        ]
    
    def extract_text_blocks(self) -> List[Dict]:
        """Extract all text blocks with formatting information"""
        blocks = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                blocks.append({
                                    'text': text,
                                    'page': page_num,
                                    'font': span["font"],
                                    'size': span["size"],
                                    'flags': span["flags"],  # Bold, italic flags
                                    'bbox': span["bbox"],
                                    'line_bbox': line["bbox"]
                                })
        
        self.text_blocks = blocks
        return blocks
    
    def is_bold_or_italic(self, flags: int) -> Tuple[bool, bool]:
        """Check if text is bold or italic based on flags"""
        is_bold = bool(flags & 16)
        is_italic = bool(flags & 2)
        return is_bold, is_italic
    
    def analyze_text_patterns(self, text: str) -> Dict[str, bool]:
        """Analyze text for heading patterns"""
        features = {
            'matches_pattern': any(re.match(pattern, text) for pattern in self.heading_patterns),
            'is_short': len(text.split()) <= 8,
            'ends_with_colon': text.endswith(':'),
            'is_all_caps': text.isupper() and len(text) > 2,
            'is_title_case': text.istitle(),
            'starts_with_number': re.match(r'^\d+\.?\s+', text) is not None,
            'has_minimal_punctuation': len(re.findall(r'[.!?]', text)) <= 1,
        }
        return features
    
    def calculate_spacing_features(self, block_idx: int) -> Dict[str, float]:
        """Calculate spacing features relative to surrounding text"""
        if not self.text_blocks or block_idx >= len(self.text_blocks):
            return {'space_before': 0, 'space_after': 0}
        
        current = self.text_blocks[block_idx]
        space_before = 0
        space_after = 0
        
        # Space before
        if block_idx > 0:
            prev_block = self.text_blocks[block_idx - 1]
            if prev_block['page'] == current['page']:
                space_before = current['line_bbox'][1] - prev_block['line_bbox'][3]
        
        # Space after
        if block_idx < len(self.text_blocks) - 1:
            next_block = self.text_blocks[block_idx + 1]
            if next_block['page'] == current['page']:
                space_after = next_block['line_bbox'][1] - current['line_bbox'][3]
        
        return {'space_before': space_before, 'space_after': space_after}
    
    def detect_headings(self, threshold: float = 0.6) -> List[Dict]:
        """Main heading detection function"""
        if not self.text_blocks:
            self.extract_text_blocks()
        
        headings = []
        
        for i, block in enumerate(self.text_blocks):
            text = block['text']
            
            if len(text) < 3 or len(text) > 200:
                continue
            is_bold, is_italic = self.is_bold_or_italic(block['flags'])
            patterns = self.analyze_text_patterns(text)
            spacing = self.calculate_spacing_features(i)
            
            # Calculate heading score
            score = 0
            if is_bold:
                score += 0.3
            if is_italic:
                score += 0.1
            
            # Pattern features (weight: 0.4)
            if patterns['matches_pattern']:
                score += 0.2
            if patterns['is_short']:
                score += 0.1
            if patterns['ends_with_colon']:
                score += 0.05
            if patterns['is_all_caps']:
                score += 0.05
            
            # Spacing features (weight: 0.2)
            avg_spacing = np.mean([s.get('space_before', 0) + s.get('space_after', 0) 
                                 for s in [self.calculate_spacing_features(j) 
                                          for j in range(max(0, i-2), min(len(self.text_blocks), i+3))]])
            
            current_spacing = spacing['space_before'] + spacing['space_after']
            if avg_spacing > 0 and current_spacing > avg_spacing * 1.2:
                score += 0.2
            
            # Position features
            line_y = block['line_bbox'][1]
            page_height = 842  
            relative_pos = line_y / page_height
            
            # Headings often appear at top of pages or after significant spacing
            if relative_pos < 0.15:  # Top of page
                score += 0.1
            
            if score >= threshold:
                headings.append({
                    'text': text,
                    'page': block['page'],
                    'score': score,
                    'is_bold': is_bold,
                    'is_italic': is_italic,
                    'patterns': patterns,
                    'bbox': block['bbox']
                })
        
        return sorted(headings, key=lambda x: (x['page'], x['bbox'][1]))
    
    def classify_heading_levels(self, headings: List[Dict]) -> List[Dict]:
        """Classify headings into hierarchical levels"""
        if not headings:
            return []
        
        # Analyze patterns to determine hierarchy
        for heading in headings:
            text = heading['text']
            level = 1  # Default level
            
            # Check for numbered patterns
            if re.match(r'^\d+\.?\s+', text):
                level = 1
            elif re.match(r'^\d+\.\d+\.?\s+', text):
                level = 2
            elif re.match(r'^\d+\.\d+\.\d+\.?\s+', text):
                level = 3
            elif re.match(r'^[A-Z]\.\s+', text):
                level = 2
            elif re.match(r'^[IVX]+\.?\s+', text):
                level = 1
            elif heading['patterns']['is_all_caps']:
                level = 1
            elif heading['is_bold'] and not heading['is_italic']:
                level = 1
            elif heading['is_bold'] and heading['is_italic']:
                level = 2
            elif heading['is_italic']:
                level = 3
            
            heading['level'] = level
        
        return headings
    
    def extract_content_under_headings(self, headings: List[Dict]) -> List[Dict]:
        """Extract text content under each heading"""
        if not headings:
            return []
        
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda x: (x['page'], x['bbox'][1]))
        
        for i, heading in enumerate(sorted_headings):
            content_blocks = []
            current_page = heading['page']
            heading_y = heading['bbox'][3]  # Bottom Y coordinate of heading
            
            # Find the next heading to know where current section ends
            next_heading_page = None
            next_heading_y = None
            
            if i + 1 < len(sorted_headings):
                next_heading = sorted_headings[i + 1]
                next_heading_page = next_heading['page']
                next_heading_y = next_heading['bbox'][1]  # Top Y coordinate of next heading
            
            # Extract content blocks that belong to this heading
            for block in self.text_blocks:
                block_page = block['page']
                block_y_top = block['line_bbox'][1]
                block_text = block['text'].strip()
                
                # Skip empty blocks and the heading itself
                if not block_text or block_text == heading['text']:
                    continue
                
                # Check if block belongs to current heading section
                belongs_to_section = False
                
                if block_page == current_page:
                    # Same page as heading
                    if block_y_top > heading_y:
                        # Block is below the heading
                        if next_heading_page == current_page:
                            # Next heading is on same page
                            if block_y_top < next_heading_y:
                                belongs_to_section = True
                        else:
                            # Next heading is on different page or doesn't exist
                            belongs_to_section = True
                
                elif block_page > current_page:
                    # Block is on a later page
                    if next_heading_page is None:
                        # No more headings, all remaining content belongs here
                        belongs_to_section = True
                    elif block_page < next_heading_page:
                        # Block is on page between current and next heading
                        belongs_to_section = True
                    elif block_page == next_heading_page and block_y_top < next_heading_y:
                        # Block is on same page as next heading but above it
                        belongs_to_section = True
                
                if belongs_to_section:
                    content_blocks.append({
                        'text': block_text,
                        'page': block_page,
                        'bbox': block['bbox'],
                        'font': block['font'],
                        'size': block['size'],
                        'flags': block['flags']
                    })
            
            # Combine content blocks into paragraphs
            heading['content'] = self.combine_blocks_into_paragraphs(content_blocks)
            heading['content_blocks'] = content_blocks
        
        return sorted_headings
    
    def combine_blocks_into_paragraphs(self, content_blocks: List[Dict]) -> List[str]:
        """Combine text blocks into coherent paragraphs"""
        if not content_blocks:
            return []
        
        # Sort blocks by page and position
        sorted_blocks = sorted(content_blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        paragraphs = []
        current_paragraph = []
        last_y = None
        last_page = None
        
        for block in sorted_blocks:
            text = block['text'].strip()
            current_y = block['bbox'][1]
            current_page = block['page']
            
            # Check if we should start a new paragraph
            start_new_paragraph = False
            
            if last_page is not None and current_page != last_page:
                # New page
                start_new_paragraph = True
            elif last_y is not None:
                # Check vertical spacing
                y_gap = abs(current_y - last_y)
                if y_gap > 20:  # Significant vertical gap suggests new paragraph
                    start_new_paragraph = True
            
            # Check for paragraph indicators
            if (text.endswith('.') or text.endswith('!') or text.endswith('?')) and len(text) > 50:
                # Likely end of sentence/paragraph
                current_paragraph.append(text)
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif start_new_paragraph:
                # Start new paragraph due to spacing
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [text]
            else:
                # Continue current paragraph
                current_paragraph.append(text)
            
            last_y = current_y
            last_page = current_page
        
        # Add remaining paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def extract_document_structure(self) -> Dict:
        """Extract complete document structure with content"""
        headings = self.detect_headings()
        headings_with_levels = self.classify_heading_levels(headings)
        headings_with_content = self.extract_content_under_headings(headings_with_levels)
        
        structure = {
            'total_pages': len(self.doc),
            'total_headings': len(headings_with_content),
            'headings_by_level': {},
            'headings': headings_with_content,
            'sections': []
        }
        
        # Group by levels and create sections
        for heading in headings_with_content:
            level = heading['level']
            if level not in structure['headings_by_level']:
                structure['headings_by_level'][level] = []
            structure['headings_by_level'][level].append(heading)
            
            # Create section entry
            section = {
                'title': heading['text'],
                'level': heading['level'],
                'page': heading['page'],
                'content_paragraphs': heading['content'],
                'word_count': sum(len(para.split()) for para in heading['content']),
                'char_count': sum(len(para) for para in heading['content'])
            }
            structure['sections'].append(section)
        
        return structure
    
    def extract_sections_list(self, file_path: str = None) -> List[Dict]:
        """Extract sections in the specified format as a list"""
        import os
        
        headings = self.detect_headings()
        headings_with_levels = self.classify_heading_levels(headings)
        headings_with_content = self.extract_content_under_headings(headings_with_levels)
        
        sections_list = []
        
        for heading in headings_with_content:
            # Combine all paragraphs into single text string
            combined_text = ' '.join(heading['content']).strip()
            
            section = {
                "document": os.path.basename(file_path) if file_path else "unknown_document.pdf",
                "page": heading["page"],
                "section_title": heading["text"].strip(),
                "text": combined_text
            }
            sections_list.append(section)
        
        return sections_list
    
    def get_section_content(self, heading_text: str) -> Dict:
        """Get content for a specific heading"""
        structure = self.extract_document_structure()
        
        for section in structure['sections']:
            if section['title'].lower() == heading_text.lower():
                return section
        
        return None
    
    def export_sections_to_text(self, output_file: str = None) -> str:
        """Export all sections to formatted text"""
        structure = self.extract_document_structure()
        output = []
        
        for section in structure['sections']:
            # Add heading with level indication
            level_prefix = '#' * section['level']
            output.append(f"{level_prefix} {section['title']}")
            output.append("")  # Empty line
            
            # Add paragraphs
            for paragraph in section['content_paragraphs']:
                output.append(paragraph)
                output.append("")  # Empty line between paragraphs
            
            output.append("=" * 50)  # Section separator
            output.append("")
        
        result = '\n'.join(output)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result

# Usage example
def main():
    file_path = "file02.pdf"
    parser = PDFHeadingParser(file_path)
    
    # Extract sections in the specified format
    sections_list = parser.extract_sections_list(file_path)
    
    print(f"Extracted {len(sections_list)} sections:")
    print("=" * 80)
    
    for i, section in enumerate(sections_list, 1):
        print(f"Section {i}:")
        print(f"Document: {section['document']}")
        print(f"Page: {section['page']+1}")  # 1-based page numbering
        print(f"Title: {section['section_title']}")
        
        # Show text preview
        text_preview = section['text'][:200] + "..." if len(section['text']) > 200 else section['text']
        print(f"Text: {text_preview}")
        print("-" * 60)
    
    return sections_list

def process_multiple_documents(file_paths: List[str]) -> List[Dict]:
    """Process multiple PDF documents and return combined sections list"""
    all_sections = []
    
    for file_path in file_paths:
        try:
            parser = PDFHeadingParser(file_path)
            sections = parser.extract_sections_list(file_path)
            all_sections.extend(sections)
            print(f"Processed {file_path}: {len(sections)} sections")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_sections


if __name__ == "__main__":
    # Install required package: pip install PyMuPDF
    sections = main()
    
    # Example of how to use the returned sections list
    print(f"\nReturned {len(sections)} sections in the specified format")
