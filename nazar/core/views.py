import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from django.core.mail import send_mail
from django.conf import settings
from django.http import JsonResponse
from ultralytics import YOLO
import threading
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect,HttpResponse,get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.contrib.auth.models import User





def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            send_welcome_email(User)
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

@login_required
def home(request):
    return render(request, 'home.html')


# Email Configuration
settings.EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
settings.EMAIL_HOST = 'smtp.gmail.com'
settings.EMAIL_PORT = 587
settings.EMAIL_USE_TLS = True
settings.EMAIL_HOST_USER = 'nazarai.info@gmail.com'
settings.EMAIL_HOST_PASSWORD = 'igyu rlba dcpi kfzp'

# Define folders
BASE_DIR = os.getcwd()
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DETECTED_IMAGES_DIR = os.path.join(BASE_DIR, "media/detected_images")

# Ensure folders exist
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(DETECTED_IMAGES_DIR, exist_ok=True)

KNOWN_DISTANCE = 50  # cm (Distance of object from camera during calibration)
KNOWN_HEIGHT = 20  # cm (Real height of the object)
FOCAL_LENGTH = None  # To be determined

# Capture an image and get the object height in pixels
PERCEIVED_HEIGHT = 150  # Example value from YOLO detection

# Calculate focal length
FOCAL_LENGTH = (PERCEIVED_HEIGHT * KNOWN_DISTANCE) / KNOWN_HEIGHT
print(f"Calculated Focal Length: {FOCAL_LENGTH}")


# Initialize YOLO model
model = YOLO(r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\models\yolov8s.pt")

# IP Webcam URL (Replace with your actual IP webcam URL)
IP_WEBCAM_URL = "http://192.0.0.4:8080/video"

# Known Faces
known_face_encodings = []
known_face_names = {
    "Yash": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\yp.jpg",
    "Khushbu": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\_MG_7750.JPG"
}

# Load Faces
for name, image_path in known_face_names.items():
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
    except Exception as e:
        print(f"Error loading {image_path}: {e}")



def send_alert(object_name, confidence_score, image_path):
    """Send an email alert instantly upon detecting a moving object."""
    def email_task():
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        email_body = (
            f"Alert! Object Detected: {object_name}\n"
            f"Confidence Score: {confidence_score:.2f}\n"
            f"Timestamp: {timestamp}\n"
            f"Check the attached image for details."
        )

        try:
            email = EmailMessage(
                subject=f" {object_name} Detected!",
                body=email_body,
                from_email=settings.EMAIL_HOST_USER,
                to=["ypgoblin@gmail.com"]
            )

            # Attach image
            with open(image_path, "rb") as img_file:
                email.attach(image_path.split("/")[-1], img_file.read(), "image/jpeg")

            email.send(fail_silently=False)
            print("Alert email sent successfully!")

        except Exception as e:
            print(f"Email failed: {e}")

    # Run email sending in a separate thread for faster execution
    threading.Thread(target=email_task).start()

def detect_objects_and_faces():
    """Enhanced motion detection with background subtraction, optical flow, and region of interest."""
    video_capture = cv2.VideoCapture(IP_WEBCAM_URL)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    previous_frame = None
    model = YOLO("models/yolov8s.pt")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            continue
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        
        ROI_TOP_LEFT = (50, 50)
        ROI_BOTTOM_RIGHT = (600, 400)
        roi = fg_mask[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
        
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(contour) > 1000 for contour in contours)
        
        if previous_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if np.mean(magnitude) > 1.5:
                motion_detected = True
        
        previous_frame = gray_frame
        
        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = model.names[int(box.cls[0])]
                
                if conf > 0.3:
                    object_height = y2 - y1
                    distance = (FOCAL_LENGTH * KNOWN_HEIGHT) / object_height if object_height > 0 else 0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}, {distance:.2f}cm", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if label == "person" and distance < 100:
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        image_path = os.path.join(DETECTED_IMAGES_DIR, f"{timestamp}.jpg")
                        cv2.imwrite(image_path, frame)
                        send_alert(f"Person Detected at {distance:.2f}cm", conf, image_path)
        
        cv2.imshow("Motion, Object & Face Detection with Distance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def start_detection(request):
    """Start object and face detection in a separate thread."""
    detection_thread = threading.Thread(target=detect_objects_and_faces)
    detection_thread.start()
    return render(request, 'live_feed.html')
    # return JsonResponse({"status": "success", "message": "Detection started."})
    
    def live_feed(request):
        """Returns a streaming response for the live feed."""
        return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def display_detected_images(request):
    """Show detected images on webpage."""
    image_list = os.listdir(DETECTED_IMAGES_DIR)
    image_urls = [f"/media/detected_images/{img}" for img in image_list]
    return render(request, "detect_images.html", {"images": image_urls})



def generate_frames():
    """Capture frames from the video feed and encode them for live streaming."""
    video_capture = cv2.VideoCapture(IP_WEBCAM_URL)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for the response stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()
def live_feed(request):
    """Returns a streaming response for the live feed."""
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_page(request):
    """Render the page with the live video feed."""
    return render(request, "live_feed.html")

# send welcome message
def send_welcome_email(user):
    """
    Sends a welcome email to the newly registered user.
    Args:
        user: The user object with email field.
    """
    subject = 'Welcome to Nazar AI'
    from_email = 'nazarai.info@gmail.com'  # Your sender email
    recipient_list = [user.email]  # User's email

    # Render HTML message using the template and user context
    html_message = render_to_string('welcome.html', {'user': user})

    # Create the email message
    email = EmailMessage(
        subject,
        html_message,
        from_email,
        recipient_list
    )

    # Set the email content type as HTML
    email.content_subtype = 'html'

    # Send the email
    email.send()

