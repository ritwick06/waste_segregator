import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# Initialize serial connection to Arduino
arduino = serial.Serial('COM3', 11600)  # Replace with your Arduino port
time.sleep(2)  # Allow connection to establish

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='C:\\Users\\Ramesh\\Desktop\\c\\projects\\waste_segregation\\training_model\\trained\\waste_classifier_quantized.tflite')
interpreter.allocate_tensors()

# Get model I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Alphabetically sorted class labels (MUST match training order!)
class_labels = [
    'aerosol_cans',
    'aluminum_food_cans',
    'aluminum_soda_cans',
    'cardboard_boxes',
    'cardboard_packaging',
    'clothing',
    'coffee_grounds',
    'disposable_plastic_cutlery',
    'eggshells',
    'food_waste',
    'glass_beverage_bottles',
    'glass_cosmetic_containers',
    'glass_food_jars',
    'magazines',
    'newspaper',
    'office_paper',
    'paper_cups',
    'plastic_cup_lids',
    'plastic_detergent_bottles',
    'plastic_food_containers',
    'plastic_shopping_bags',
    'plastic_soda_bottles',
    'plastic_straws',
    'plastic_trash_bags',
    'plastic_water_bottles',
    'shoes',
    'steel_food_cans',
    'styrofoam_cups',
    'styrofoam_food_containers',
    'tea_bags'
]

# Material-based angle mapping (0-180°)
ANGLE_MAPPING = {
    # Plastics (30°)
    'plastic_water_bottles': 30,
    'plastic_soda_bottles': 30,
    'plastic_detergent_bottles': 30,
    'plastic_food_containers': 30,
    'plastic_cup_lids': 30,
    'plastic_shopping_bags': 30,
    'plastic_straws': 30,
    'plastic_trash_bags': 30,
    'disposable_plastic_cutlery': 30,
    
    # Metals (60°)
    'aluminum_soda_cans': 60,
    'aluminum_food_cans': 60,
    'steel_food_cans': 60,
    'aerosol_cans': 60,
    
    # Glass (90°)
    'glass_beverage_bottles': 90,
    'glass_cosmetic_containers': 90,
    'glass_food_jars': 90,
    
    # Paper/Cardboard (120°)
    'cardboard_boxes': 120,
    'cardboard_packaging': 120,
    'paper_cups': 120,
    'magazines': 120,
    'newspaper': 120,
    'office_paper': 120,
    
    # Organic (150°)
    'food_waste': 150,
    'coffee_grounds': 150,
    'eggshells': 150,
    'tea_bags': 150,
    
    # Textiles/Styrofoam/Other (180°)
    'clothing': 180,
    'shoes': 180,
    'styrofoam_cups': 180,
    'styrofoam_food_containers': 180
}

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Get predicted class
    class_idx = np.argmax(predictions)
    detected_class = class_labels[class_idx]
    confidence = np.max(predictions) * 100

    # Only act on high-confidence predictions
    if confidence > 75:
        angle = ANGLE_MAPPING.get(detected_class, 180)  # Default to 180°
        arduino.write(f"{angle}\n".encode())
        print(f"Detected: {detected_class} -> Angle: {angle}°")

    # Display info
    cv2.putText(frame, 
                f"{detected_class} ({confidence:.1f}%)", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    cv2.imshow('Waste Sorting System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
arduino.close()
cap.release()
cv2.destroyAllWindows()