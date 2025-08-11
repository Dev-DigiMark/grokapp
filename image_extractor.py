import numpy as np
import io, cv2, pytesseract
from ultralytics import YOLO
import pyzbar.pyzbar as pyzbar
from PIL import Image, ExifTags

def extract_image_info(image_input, detect_objects=False):
    """
    Extract comprehensive information from an image.
    
    Args:
        image_input: File-like object or bytes containing the image
        detect_objects (bool): Whether to perform object detection with YOLO
        
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
            "mode": img.mode
        }
        
        # EXIF data
        try:
            exif_data = img._getexif()
            if exif_data:
                info["exif"] = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
        except Exception:
            info["exif"] = None
            
        # Convert to OpenCV format for further processing
        img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # OCR with Tesseract
        try:
            ocr_text = pytesseract.image_to_string(img_cv)
            info["text"] = ocr_text.strip() if ocr_text else ""
        except Exception:
            info["text"] = ""
            
        # Barcode detection
        try:
            barcodes = pyzbar.decode(img_cv)
            info["barcodes"] = [{"type": b.type, "data": b.data.decode('utf-8')} for b in barcodes]
        except Exception:
            info["barcodes"] = []
            
        # Face detection
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            info["faces"] = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
        except Exception:
            info["faces"] = []
            
        # Dominant colors
        try:
            pixels = np.float32(img_cv.reshape(-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, 5, None, criteria, 10, flags)
            info["dominant_colors"] = [palette[i].tolist() for i in range(len(palette))]
        except Exception:
            info["dominant_colors"] = []
            
        # Object detection with YOLO (optional)
        if detect_objects:
            try:
                model = YOLO('yolov8n.pt')  # Use nano model for speed
                results = model(img_cv)
                objects = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        label = model.names[cls_id]
                        conf = float(box.conf)
                        xyxy = box.xyxy[0].tolist()
                        objects.append({
                            "class": label,
                            "confidence": conf,
                            "bbox": {"x_min": xyxy[0], "y_min": xyxy[1], "x_max": xyxy[2], "y_max": xyxy[3]}
                        })
                info["objects"] = objects
            except Exception:
                info["objects"] = []
        else:
            info["objects"] = []
            
        return info
        
    except Exception as e:
        return {"error": str(e)}