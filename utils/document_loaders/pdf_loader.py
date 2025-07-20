# utils/document_loaders/pdf_loader.py

from typing import List, Iterator, Union, Optional
import io
import logging
from langchain.schema import Document
import fitz
from .base import BaseDocumentLoader

logger = logging.getLogger(__name__)

# ——— Basic Class ————————————————————————————————

class PDFLoader(BaseDocumentLoader):
    """
    Optimized PDF loader with performance enhancements for large documents
    """
    
    def __init__(self, 
                extract_images: bool = False,
                min_page_length: int = 50,
                text_flags: int = None):
        """
        Args:
            extract_images: Whether to extract images (slower)
            min_page_length: Minimum characters to consider a page valid
            text_flags: PyMuPDF text extraction flags for optimization
        """
        self.extract_images = extract_images
        self.min_page_length = min_page_length
        
        # Optimized text extraction flags
        if text_flags is None:
            # Remove whitespace preservation and ligatures for speed
            self.text_flags = (
                fitz.TEXTFLAGS_TEXT & 
                ~fitz.TEXT_PRESERVE_WHITESPACE & 
                ~fitz.TEXT_PRESERVE_LIGATURES
            )
        else:
            self.text_flags = text_flags

    def load_documents(self, source: Union[str, io.BytesIO]) -> List[Document]:
        """Load all documents into memory (less efficient, but kept for compatibility)"""
        docs = []
        for doc in self.stream_documents(source):
            docs.append(doc)
        return docs

    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        """
        Optimized streaming with early page validation and faster text extraction
        """
        # Ensure we're at the start of the stream
        if isinstance(source, io.BytesIO):
            source.seek(0)
            
        try:
            # Open PDF with minimal overhead
            pdf = (fitz.open(stream=source, filetype="pdf") 
                if isinstance(source, io.BytesIO) 
                else fitz.open(source))
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Cannot open PDF: {e}")
        
        try:
            total_pages = len(pdf)
            logger.debug(f"Processing PDF with {total_pages} pages")
            
            for page_num in range(total_pages):
                try:
                    page = pdf[page_num]
                    
                    # Fast text extraction with optimized flags
                    text = page.get_text("text", flags=self.text_flags)
                    
                    # Skip nearly empty pages early
                    if len(text.strip()) < self.min_page_length:
                        logger.debug(f"Skipping page {page_num + 1}: too short ({len(text)} chars)")
                        continue
                    
                    # Create metadata with useful info
                    metadata = {
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "page_length": len(text),
                        "source_type": "pdf_stream" if isinstance(source, io.BytesIO) else "pdf_file"
                    }
                    
                    # Add page dimensions if needed (minimal overhead)
                    if page.rect:
                        metadata.update({
                            "page_width": round(page.rect.width, 1),
                            "page_height": round(page.rect.height, 1)
                        })
                    
                    yield Document(page_content=text, metadata=metadata)
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
                    
        finally:
            pdf.close()

    def get_document_info(self, source: Union[str, io.BytesIO]) -> dict:
        """
        Quick document analysis without full text extraction
        """
        if isinstance(source, io.BytesIO):
            source.seek(0)
            
        try:
            pdf = (fitz.open(stream=source, filetype="pdf") 
                if isinstance(source, io.BytesIO) 
                else fitz.open(source))
        except Exception as e:
            return {"error": str(e)}
        
        try:
            info = {
                "page_count": len(pdf),
                "file_size": pdf.stream_size if hasattr(pdf, 'stream_size') else None,
                "metadata": pdf.metadata,
                "is_encrypted": pdf.needs_pass,
                "estimated_text_pages": 0
            }
            
            # Quick sample to estimate text content (check first 3 pages)
            sample_pages = min(3, len(pdf))
            text_chars = 0
            
            for i in range(sample_pages):
                try:
                    page_text = pdf[i].get_text("text", flags=self.text_flags)
                    if len(page_text.strip()) >= self.min_page_length:
                        info["estimated_text_pages"] += 1
                        text_chars += len(page_text)
                except:
                    continue
            
            # Estimate total text pages
            if sample_pages > 0:
                info["estimated_text_pages"] = int(
                    (info["estimated_text_pages"] / sample_pages) * info["page_count"]
                )
                info["estimated_chars_per_page"] = text_chars // max(sample_pages, 1)
            
            return info
            
        finally:
            pdf.close()

# ——— HighPerformance Class ————————————————————————————————

class HighPerformancePDFLoader(PDFLoader):
    """
    Ultra-high performance PDF loader for large documents
    Uses more aggressive optimizations
    """
    
    def __init__(self, 
                batch_pages: int = 10,
                skip_images: bool = True,
                fast_text_only: bool = True):
        """
        Args:
            batch_pages: Process pages in batches for better memory management
            skip_images: Skip image-heavy pages entirely
            fast_text_only: Use fastest possible text extraction
        """
        # Ultra-fast text extraction flags
        text_flags = fitz.TEXT_PRESERVE_IMAGES if not skip_images else 0
        
        super().__init__(
            extract_images=False,
            min_page_length=20,  # Lower threshold for speed
            text_flags=text_flags
        )
        
        self.batch_pages = batch_pages
        self.skip_images = skip_images
        self.fast_text_only = fast_text_only

    def stream_documents(self, source: Union[str, io.BytesIO]) -> Iterator[Document]:
        """
        Streaming with batch processing and aggressive optimizations
        """
        if isinstance(source, io.BytesIO):
            source.seek(0)
            
        try:
            pdf = (fitz.open(stream=source, filetype="pdf") 
                if isinstance(source, io.BytesIO) 
                else fitz.open(source))
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
        
        try:
            total_pages = len(pdf)
            
            # Process in batches for better memory management
            for batch_start in range(0, total_pages, self.batch_pages):
                batch_end = min(batch_start + self.batch_pages, total_pages)
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = pdf[page_num]
                        
                        # Skip image-heavy pages if requested
                        if self.skip_images and self._is_image_heavy(page):
                            logger.debug(f"Skipping image-heavy page {page_num + 1}")
                            continue
                        
                        # Ultra-fast text extraction
                        if self.fast_text_only:
                            text = page.get_text()  # Fastest method
                        else:
                            text = page.get_text("text", flags=self.text_flags)
                        
                        if len(text.strip()) < self.min_page_length:
                            continue
                        
                        metadata = {
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "batch_start": batch_start + 1,
                            "batch_end": batch_end
                        }
                        
                        yield Document(page_content=text, metadata=metadata)
                        
                    except Exception as e:
                        logger.warning(f"Error in batch processing page {page_num + 1}: {e}")
                        continue
                        
        finally:
            pdf.close()
    
    def _is_image_heavy(self, page) -> bool:
        """Quick check if page is image-heavy"""
        try:
            # Simple heuristic: if text/area ratio is very low, likely image-heavy
            text_length = len(page.get_text())
            page_area = page.rect.width * page.rect.height
            
            if page_area > 0:
                text_density = text_length / page_area
                return text_density < 0.01  # Adjust threshold as needed
            
            return False
        except:
            return False