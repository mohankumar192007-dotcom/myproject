import cv2
import pyttsx3
import time
from ultralytics import YOLO

# ==========================================
# 1. SETUP & CONFIGURATION (DroidCam IP)
# ==========================================
PHONE_IP = "192.168.29.250:4747"
URL = f"http://{PHONE_IP}/video"

# Load lightweight YOLOv8 nano model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Voice Alert Settings
last_speak_time = 0
SPEAK_INTERVAL = 3  # 3 seconds gap between alerts
last_object = ""     # To keep track of previous object

print(f"Connecting to Phone Camera at {URL}...")
cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("❌ Error: Could not connect to DroidCam!")
    exit()

print("✅ Connected successfully! Press 'q' to stop.")

# Helper function to speak without freezing
def speak_msg(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech Error: {e}")

# ==========================================
# 2. MAIN REAL-TIME DETECTION LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLOv8 detection
    results = model(frame, conf=0.5, verbose=False)

    detected_objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_objects.append(class_name)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Smart Audio Logic
    current_time = time.time()
    
    if detected_objects:
        primary_object = detected_objects[0]

        # Conditions to speak:
        # 1. New object detect aana udanae pesum (OR)
        # 2. Same object-a irundha 3 seconds gap apram repeat pesum
        if (primary_object != last_object) or (current_time - last_speak_time > SPEAK_INTERVAL):
            alert_msg = f"Ahead: {primary_object}"
            print(f"🔊 Audio Alert: {alert_msg}")
            
            speak_msg(alert_msg)
            
            last_speak_time = current_time
            last_object = primary_object

    cv2.imshow("Mohan's Digital Eye - Phone Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped successfully.")