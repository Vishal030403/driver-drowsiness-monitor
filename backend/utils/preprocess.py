import cv2
import numpy as np
from typing import Tuple, Optional

def detect_eyes(face_image):
    """
    Detect eyes in a face image using Haar Cascade
    Returns: tuple of (left_eye, right_eye) or (None, None) if not detected
    """
    try:
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda x: x[0])
            return eyes[0], eyes[1]
        
        return None, None
    
    except Exception as e:
        print(f"Error detecting eyes: {e}")
        return None, None

def calculate_eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) for drowsiness detection
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # This is a simplified version - implement based on your needs
    pass

def enhance_image_quality(image):
    """
    Enhance image quality for better detection
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def draw_detection_box(frame, face_coords, status, confidence):
    """
    Draw bounding box and status on frame
    """
    if face_coords is None:
        return frame
    
    x, y, w, h = face_coords
    
    # Choose color based on status
    color = (0, 0, 255) if status == "drowsy" else (0, 255, 0)  # Red for drowsy, Green for alert
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw status text
    text = f"{status.upper()} ({confidence:.2f})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image
    """
    import base64
    
    # Remove header if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image to base64 string
    """
    import base64
    
    # Encode image
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"