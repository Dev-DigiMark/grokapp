import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from typing import Dict, Any, Optional

def extract_image_info_lightweight(image_input, enhance_ocr: bool = True) -> Dict[str, Any]:
    """
    Lightweight image information extractor using only PIL and pytesseract.
    Much faster than the full version but with limited capabilities.
    
    Args:
        image_input: File-like object or bytes containing the image
        enhance_ocr (bool): Whether to enhance image for better OCR results
        
    Returns:
        dict: Dictionary containing extracted image information
    """
    try:
        # Handle input type
        if hasattr(image_input, 'read'):
            image_bytes = image_input.read()
        else:
            image_bytes = image_input
            
        # Open image with PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Basic properties
        info = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_mb": len(image_bytes) / (1024 * 1024)
        }
        
        # EXIF data (lightweight)
        try:
            exif_data = img._getexif()
            if exif_data:
                # Only extract basic EXIF info to avoid heavy processing
                basic_exif = {}
                for tag_id, value in exif_data.items():
                    if tag_id in [271, 272, 306, 36867, 36868]:  # Make, Model, DateTime, DateTimeOriginal, DateTimeDigitized
                        basic_exif[tag_id] = str(value)
                info["exif_basic"] = basic_exif
        except Exception:
            info["exif_basic"] = {}
        
        # Simple image analysis without OpenCV
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img_rgb = img.convert('RGB')
            else:
                img_rgb = img
            
            # Get basic color information
            colors = img_rgb.getcolors(maxcolors=1000)
            if colors:
                # Sort by frequency and get top 5 colors
                sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                info["dominant_colors"] = [{"count": count, "rgb": rgb} for count, rgb in sorted_colors]
            
            # Simple brightness analysis
            enhancer = ImageEnhance.Brightness(img_rgb)
            brightness_factor = enhancer.enhance(1.0)
            # Get average brightness from a sample of pixels
            sample_pixels = []
            step = max(1, min(img.width, img.height) // 20)  # Sample every 20th pixel
            for y in range(0, img.height, step):
                for x in range(0, img.width, step):
                    sample_pixels.append(img_rgb.getpixel((x, y)))
            
            if sample_pixels:
                avg_brightness = sum(sum(pixel) / 3 for pixel in sample_pixels) / len(sample_pixels)
                info["brightness_analysis"] = {
                    "average_brightness": round(avg_brightness, 2),
                    "brightness_level": "dark" if avg_brightness < 85 else "normal" if avg_brightness < 170 else "bright"
                }
                
        except Exception as e:
            info["image_analysis_error"] = str(e)
        
        # OCR with pytesseract (the main feature)
        try:
            if enhance_ocr:
                # Create enhanced version for better OCR
                enhanced_img = enhance_image_for_ocr(img)
                ocr_text = pytesseract.image_to_string(enhanced_img, config='--psm 6')
            else:
                ocr_text = pytesseract.image_to_string(img, config='--psm 6')
            
            info["text"] = ocr_text.strip() if ocr_text else ""
            
            # Basic text analysis
            if info["text"]:
                lines = info["text"].split('\n')
                info["text_analysis"] = {
                    "line_count": len(lines),
                    "word_count": len(info["text"].split()),
                    "character_count": len(info["text"]),
                    "has_numbers": any(char.isdigit() for char in info["text"]),
                    "has_uppercase": any(char.isupper() for char in info["text"])
                }
                
                # Look for common patterns (driver's license, ID card, etc.)
                text_lower = info["text"].lower()
                if any(keyword in text_lower for keyword in ['driver', 'license', 'id', 'card', 'passport']):
                    info["document_type"] = "likely_identification"
                elif any(keyword in text_lower for keyword in ['police', 'report', 'accident', 'incident']):
                    info["document_type"] = "likely_police_report"
                elif any(keyword in text_lower for keyword in ['medical', 'hospital', 'doctor', 'treatment']):
                    info["document_type"] = "likely_medical_document"
                else:
                    info["document_type"] = "unknown"
                    
        except Exception as e:
            info["text"] = ""
            info["ocr_error"] = str(e)
        
        # Simple barcode detection using text patterns (no pyzbar)
        try:
            if info["text"]:
                # Look for common barcode patterns in text
                barcode_patterns = []
                lines = info["text"].split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for lines with only numbers/letters and consistent length
                    if line and len(line) >= 8 and len(line) <= 50:
                        if all(c.isalnum() or c in '-./ ' for c in line):
                            # Check if it looks like a barcode pattern
                            if len(set(line)) > 5:  # Has variety of characters
                                barcode_patterns.append({
                                    "type": "potential_barcode",
                                    "data": line,
                                    "confidence": "low"
                                })
                
                info["potential_barcodes"] = barcode_patterns
            else:
                info["potential_barcodes"] = []
                
        except Exception as e:
            info["potential_barcodes"] = []
            info["barcode_error"] = str(e)
        
        # Set default values for missing features
        info["barcodes"] = []  # No pyzbar
        info["faces"] = []     # No OpenCV face detection
        info["objects"] = []   # No YOLO
        info["warning"] = "Lightweight mode - limited features but faster processing"
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

def enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhance image for better OCR results using PIL only.
    """
    try:
        # Convert to grayscale for better OCR
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_gray)
        img_contrast = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img_contrast)
        img_sharp = enhancer.enhance(1.2)
        
        # Apply slight blur to reduce noise
        img_enhanced = img_sharp.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img_enhanced
        
    except Exception:
        # If enhancement fails, return original
        return img

def extract_text_only(image_input) -> str:
    """
    Extract only text from image - fastest method.
    """
    try:
        if hasattr(image_input, 'read'):
            image_bytes = image_input.read()
        else:
            image_bytes = image_input
            
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip() if text else ""
        
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_image_metadata(image_input) -> Dict[str, Any]:
    """
    Get only basic image metadata - very fast.
    """
    try:
        if hasattr(image_input, 'read'):
            image_bytes = image_input.read()
        else:
            image_bytes = image_input
            
        img = Image.open(io.BytesIO(image_bytes))
        
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_mb": round(len(image_bytes) / (1024 * 1024), 2)
        }
        
    except Exception as e:
        return {"error": str(e)}
