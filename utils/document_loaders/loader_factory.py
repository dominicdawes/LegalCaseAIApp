# utils/document_loaders/loader_factory.py

import os
import io
import logging
from typing import Type, Union, Optional, Dict, Any
from .base import BaseDocumentLoader
from .pdf_loader import PDFLoader, HighPerformancePDFLoader
from .pdf_ocr_loader import PDFOCRLoader
from .docx_loader import DocxLoader
from .epub_loader import EpubLoader

logger = logging.getLogger(__name__)

# Map extension→loader class remains the same.
_LOADER_MAP: dict[str, Type[BaseDocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    # .doc is intentionally removed. See explanation below.
    ".epub": EpubLoader,
    # you could add ".txt": TxtLoader, etc.
    # you could add ".md": MarkdownLoader, etc.
}

# Performance-optimized loaders for specific use cases
_HIGH_PERFORMANCE_MAP: Dict[str, Type[BaseDocumentLoader]] = {
    ".pdf": HighPerformancePDFLoader,
    # Add more high-performance variants
}

# ——— DocumentAnalyzer Class ————————————————————————————————

class DocumentAnalyzer:
    """Analyzes documents to determine optimal processing strategy"""
    
    @staticmethod
    def analyze_pdf_performance(file_stream: io.BytesIO) -> Dict[str, Any]:
        """
        Analyze PDF characteristics to choose optimal processing strategy
        """
        import fitz
        
        file_stream.seek(0)
        analysis = {
            "file_size_mb": len(file_stream.getvalue()) / (1024 * 1024),
            "is_text_based": False,
            "estimated_pages": 0,
            "complexity_score": 0,
            "recommended_loader": "standard",
            "processing_time_estimate": 0
        }
        
        try:
            pdf = fitz.open(stream=file_stream, filetype="pdf")
            analysis["estimated_pages"] = len(pdf)
            
            # Sample first few pages for analysis
            sample_pages = min(3, len(pdf))
            total_chars = 0
            image_count = 0
            
            for i in range(sample_pages):
                try:
                    page = pdf[i]
                    text = page.get_text()
                    total_chars += len(text)
                    
                    # Count images (simplified)
                    image_list = page.get_images()
                    image_count += len(image_list)
                    
                except Exception:
                    continue
            
            # Determine if text-based
            chars_per_page = total_chars / max(sample_pages, 1)
            analysis["is_text_based"] = chars_per_page >= 100
            analysis["avg_chars_per_page"] = chars_per_page
            analysis["images_per_page"] = image_count / max(sample_pages, 1)
            
            # Calculate complexity score (0-10)
            complexity = 0
            if analysis["file_size_mb"] > 10:
                complexity += 2
            if analysis["estimated_pages"] > 100:
                complexity += 2
            if analysis["images_per_page"] > 2:
                complexity += 3
            if chars_per_page < 500:  # Likely image-heavy or formatted
                complexity += 3
                
            analysis["complexity_score"] = min(complexity, 10)
            
            # Recommend processing strategy
            if analysis["complexity_score"] >= 7:
                analysis["recommended_loader"] = "high_performance"
                analysis["processing_time_estimate"] = analysis["estimated_pages"] * 0.1  # 0.1s per page
            elif not analysis["is_text_based"]:
                analysis["recommended_loader"] = "ocr"
                analysis["processing_time_estimate"] = analysis["estimated_pages"] * 2.0  # 2s per page for OCR
            else:
                analysis["recommended_loader"] = "standard"
                analysis["processing_time_estimate"] = analysis["estimated_pages"] * 0.05  # 0.05s per page
                
            pdf.close()
            
        except Exception as e:
            logger.warning(f"PDF analysis failed: {e}")
            analysis["error"] = str(e)
            analysis["recommended_loader"] = "standard"
        
        finally:
            file_stream.seek(0)
            
        return analysis

def is_pdf_text_based(file_stream: io.BytesIO, min_char_threshold: int = 100) -> bool:
    """
    Optimized PDF text detection with early exit
    """
    import fitz
    
    file_stream.seek(0)
    
    try:
        pdf = fitz.open(stream=file_stream, filetype="pdf")
    except Exception:
        return False
    
    try:
        extracted_chars = 0
        # Check only first 3 pages for speed
        for page_num in range(min(3, len(pdf))):
            try:
                page = pdf[page_num]
                text = page.get_text()
                extracted_chars += len(text.strip())
                
                # Early exit if we have enough text
                if extracted_chars >= min_char_threshold:
                    return True
                    
            except Exception:
                continue
                
        return extracted_chars >= min_char_threshold
        
    finally:
        pdf.close()
        file_stream.seek(0)

def get_loader_for(filename: str, 
                    file_like_object: io.BytesIO, 
                    performance_mode: str = "auto",
                    analyze_first: bool = True) -> BaseDocumentLoader:
    """
    Enhanced loader factory with performance optimization
    
    Args:
        filename: Original filename for extension detection
        file_like_object: In-memory file content
        performance_mode: "auto", "standard", "high_performance", "ocr"
        analyze_first: Whether to analyze document before choosing loader
    """
    ext = os.path.splitext(filename.lower())[1]
    
    if ext not in _LOADER_MAP:
        raise ValueError(
            f"Unsupported document type '{ext}'. Supported: {list(_LOADER_MAP.keys())}"
        )
    
    # PDF-specific logic with performance optimization
    if ext == ".pdf":
        file_like_object.seek(0)
        
        # Analyze document if requested
        if analyze_first and performance_mode == "auto":
            analysis = DocumentAnalyzer.analyze_pdf_performance(file_like_object)
            
            logger.info(f"PDF Analysis: {analysis['file_size_mb']:.1f}MB, "
                        f"{analysis['estimated_pages']} pages, "
                        f"complexity: {analysis['complexity_score']}/10, "
                        f"recommended: {analysis['recommended_loader']}")
            
            performance_mode = analysis["recommended_loader"]
        
        # Choose appropriate loader based on analysis or explicit mode
        if performance_mode == "high_performance":
            loader = HighPerformancePDFLoader(
                batch_pages=20,  # Larger batches for big docs
                skip_images=True,
                fast_text_only=True
            )
        elif performance_mode == "ocr":
            loader = PDFOCRLoader()
        elif performance_mode == "auto" or performance_mode == "standard":
            # Fallback to text-based detection
            if is_pdf_text_based(file_like_object):
                loader = PDFLoader(
                    min_page_length=50,
                    extract_images=False
                )
            else:
                loader = PDFOCRLoader()
        else:
            # Default standard loader
            loader = PDFLoader()
        
        file_like_object.seek(0)
        return loader
    
    # For non-PDF files, use standard loaders
    LoaderCls = _LOADER_MAP[ext]
    return LoaderCls()

def get_high_performance_loader(filename: str, file_like_object: io.BytesIO) -> BaseDocumentLoader:
    """
    Convenience function to get the fastest possible loader
    """
    return get_loader_for(
        filename, 
        file_like_object, 
        performance_mode="high_performance",
        analyze_first=False
    )

def analyze_document_before_processing(filename: str, file_like_object: io.BytesIO) -> Dict[str, Any]:
    """
    Analyze document and return processing recommendations
    """
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == ".pdf":
        return DocumentAnalyzer.analyze_pdf_performance(file_like_object)
    else:
        # Basic analysis for other file types
        file_size_mb = len(file_like_object.getvalue()) / (1024 * 1024)
        return {
            "file_type": ext,
            "file_size_mb": file_size_mb,
            "processing_time_estimate": file_size_mb * 0.5,  # Rough estimate
            "recommended_loader": "standard"
        }