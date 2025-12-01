import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrowsinessDetector:
    def __init__(self, model_path: str):
        """Initialize the drowsiness detector with trained model"""
        self.model_path = "model\driver_drowsiness_final_model.keras"
        self.model = None
        self.img_size = (224, 224)  # Model input size
        
        # Initialize face detection cascades
        self.face_cascade = None
        self.face_cascade_alt = None
        
        # Load model and cascades
        self.load_model()
        self.load_face_detectors()
        
    def load_face_detectors(self):
        """Load Haar Cascade face detectors"""
        try:
            # Primary face detector (most accurate)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Alternative detector (more sensitive, backup)
            self.face_cascade_alt = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            )
            
            logger.info("✅ Face detection cascades loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading face detectors: {e}")
            raise
        
    def load_model(self):
        """Load the trained Keras model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            logger.info(f"📊 Model input shape: {self.model.input_shape}")
            logger.info(f"📊 Model output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def detect_and_extract_face(self, frame):
        """
        STEP 1: Detect face in frame and extract ONLY the face region
        
        Args:
            frame: Full camera frame (BGR format from OpenCV)
            
        Returns:
            face_image: Cropped face region
            face_coords: (x, y, w, h) coordinates of face
            annotated_frame: Frame with green rectangle around face (for visualization)
        """
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            # Try primary detector with RELAXED settings
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no face found, try alternative detector
            if len(faces) == 0:
                faces = self.face_cascade_alt.detectMultiScale(
                    gray,
                    scaleFactor=1.03,
                    minNeighbors=2,
                    minSize=(50, 50)
                )
            
            # If still no face, try VERY relaxed settings
            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.02,
                    minNeighbors=1,
                    minSize=(40, 40)
                )
            
            if len(faces) > 0:
                # Get the largest face (closest to camera)
                faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces_sorted[0]
                
                # Add 15% padding around face
                padding_x = int(w * 0.15)
                padding_y = int(h * 0.15)
                
                x_start = max(0, x - padding_x)
                y_start = max(0, y - padding_y)
                x_end = min(frame.shape[1], x + w + padding_x)
                y_end = min(frame.shape[0], y + h + padding_y)
                
                # Extract face region
                face_image = frame[y_start:y_end, x_start:x_end]
                
                # Verify extracted face is valid
                if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                    logger.warning(f"⚠️ Detected face too small: {face_image.shape}")
                    return None, None, frame
                
                # Create annotated frame with green rectangle
                annotated_frame = frame.copy()
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                logger.debug(f"✅ Face detected: {w}x{h} at ({x}, {y})")
                
                return face_image, (x, y, w, h), annotated_frame
            
            else:
                logger.debug("❌ No face detected in frame")
                return None, None, frame
        
        except Exception as e:
            logger.error(f"❌ Error detecting face: {e}")
            return None, None, frame
    
    def preprocess_face_for_model(self, face_image):
        """
        STEP 2: Preprocess ONLY the face image for model prediction
        
        Args:
            face_image: Cropped face region (BGR format)
            
        Returns:
            Preprocessed array ready for model (1, 224, 224, 3)
        """
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Convert to PIL Image for resizing
            pil_image = Image.fromarray(face_rgb)
            
            # Resize to model input size (224x224)
            pil_image = pil_image.resize(self.img_size, Image.LANCZOS)
            
            # Convert back to numpy array
            img_array = np.array(pil_image)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype('float32') / 255.0
            
            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.debug(f"✅ Face preprocessed: {img_array.shape}")
            
            return img_array
        
        except Exception as e:
            logger.error(f"❌ Error preprocessing face: {e}")
            raise
    
    def predict_drowsiness(self, preprocessed_face):
        """
        STEP 3: Make prediction using the model
        
        Args:
            preprocessed_face: Preprocessed face array (1, 224, 224, 3)
            
        Returns:
            dict with prediction results
        """
        try:
            # Make prediction
            prediction = self.model.predict(preprocessed_face, verbose=0)
            pred_array = prediction[0]
            
            logger.debug(f"📊 Raw model output: {pred_array}")
            
            # Handle different model output formats
            if len(pred_array) == 2:
                # Binary classification: [alert_prob, drowsy_prob]
                alert_prob = float(pred_array[0])
                drowsy_prob = float(pred_array[1])
                
                if drowsy_prob > alert_prob:
                    status = "drowsy"
                    confidence = drowsy_prob
                    predicted_class = 1
                else:
                    status = "alert"
                    confidence = alert_prob
                    predicted_class = 0
                    
            elif len(pred_array) == 1:
                # Single output: sigmoid activation
                prob = float(pred_array[0])
                
                if prob > 0.5:
                    status = "drowsy"
                    confidence = prob
                    predicted_class = 1
                else:
                    status = "alert"
                    confidence = 1 - prob
                    predicted_class = 0
            else:
                # Multi-class or unexpected format
                predicted_class = int(np.argmax(pred_array))
                confidence = float(np.max(pred_array))
                status = "drowsy" if predicted_class == 1 else "alert"
            
            # Ensure confidence is reasonable (minimum 50% for better UX)
            if confidence < 0.5:
                confidence = 0.5
            
            # Cap confidence at 98% (more realistic)
            if confidence > 0.98:
                confidence = 0.98
            
            result = {
                "status": status,
                "confidence": float(confidence),
                "predicted_class": int(predicted_class),
                "raw_prediction": pred_array.tolist()
            }
            
            logger.info(f"🎯 Prediction: {status.upper()} | Confidence: {confidence:.2%}")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise
    
    def predict_from_frame(self, frame):
        """
        MAIN FUNCTION: Complete pipeline from frame to prediction
        
        Flow:
        1. Detect face in full frame
        2. Extract ONLY the face region
        3. Preprocess face for model
        4. Make prediction
        
        Args:
            frame: Full camera frame (BGR format)
            
        Returns:
            dict with prediction results and face coordinates
        """
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                logger.warning("⚠️ Received empty frame")
                return {
                    "status": "no_face",
                    "confidence": 0.0,
                    "message": "Invalid frame",
                    "face_coords": None
                }
            
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                logger.warning(f"⚠️ Frame too small: {frame.shape}")
                return {
                    "status": "no_face",
                    "confidence": 0.0,
                    "message": "Frame too small",
                    "face_coords": None
                }
            
            # STEP 1: Detect and extract face
            face_image, face_coords, annotated_frame = self.detect_and_extract_face(frame)
            
            if face_image is None:
                return {
                    "status": "no_face",
                    "confidence": 0.0,
                    "message": "No face detected in frame",
                    "face_coords": None
                }
            
            # STEP 2: Preprocess the extracted face
            preprocessed_face = self.preprocess_face_for_model(face_image)
            
            # STEP 3: Make prediction on the face
            result = self.predict_drowsiness(preprocessed_face)
            
            # Add face coordinates to result
            result["face_coords"] = face_coords
            result["message"] = "Prediction successful"
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Error in predict_from_frame: {e}")
            return {
                "status": "error",
                "confidence": 0.0,
                "message": str(e),
                "face_coords": None
            }
    
    def test_detection(self, frame):
        """
        Debug function: Test face detection and return annotated frame
        
        Returns:
            annotated_frame: Frame with green rectangle around detected face
            detection_info: Dict with detection details
        """
        try:
            face_image, coords, annotated_frame = self.detect_and_extract_face(frame)
            
            if coords is not None:
                x, y, w, h = coords
                detection_info = {
                    "face_detected": True,
                    "coordinates": coords,
                    "face_size": (w, h),
                    "frame_size": (frame.shape[1], frame.shape[0]),
                    "face_area_percentage": (w * h) / (frame.shape[0] * frame.shape[1]) * 100
                }
            else:
                cv2.putText(annotated_frame, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                detection_info = {
                    "face_detected": False,
                    "coordinates": None,
                    "face_size": None,
                    "frame_size": (frame.shape[1], frame.shape[0]),
                    "face_area_percentage": 0
                }
            
            return annotated_frame, detection_info
        
        except Exception as e:
            logger.error(f"❌ Error in test_detection: {e}")
            return frame, {"error": str(e)}