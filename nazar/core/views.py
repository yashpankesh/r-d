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
            user = form.save()  # Save the user object
            send_welcome_email(user)  # Pass the actual user object here
            
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')  # Redirect to login page after registration
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

# Modify the send_welcome_email function to ensure it works properly.
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

# Initialize YOLO model
model = YOLO(r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\models\yolov8s.pt")

# IP Webcam URL (Replace with your actual IP webcam URL)
IP_WEBCAM_URL = "http://10.61.88.10:8080/video"

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
    """Send email with the detected image as an attachment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    email_body = (
        f"Object Detected: {object_name}\n"
        f"Confidence Score: {confidence_score:.2f}\n"
        f"Timestamp: {timestamp}\n"
        f"Check the attached image for details."
    )

    try:
        # Create email message
        email = EmailMessage(
            subject=f"ALERT: {object_name} Detected!",
            body=email_body,
            from_email=settings.EMAIL_HOST_USER,
            to=["yash.pankesh@gmail.com"]
        )

        # Attach the image file
        with open(image_path, "rb") as img_file:
            email.attach(image_path.split("/")[-1], img_file.read(), "image/jpeg")

        # Send email
        email.send(fail_silently=False)
        print("Email with image sent successfully!")

    except Exception as e:
        print(f"Email failed: {e}")
def detect_objects_and_faces():
    """Process video feed from IP webcam for motion, object, and face detection."""
    video_capture = cv2.VideoCapture(IP_WEBCAM_URL)
    
    # Initialize motion detection variables
    previous_frame = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Motion Detection
        motion_detected = False
        if previous_frame is None:
            previous_frame = gray_frame
            continue

        frame_delta = cv2.absdiff(previous_frame, gray_frame)
        threshold_frame = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
        contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                motion_detected = True
                break
        
        previous_frame = gray_frame  # Update previous frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = list(known_face_names.keys())[best_match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            detected_faces.append(name)

        # Object Detection
        results = model(frame)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = model.names[int(box.cls[0])]

                if conf > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    detected_objects.append(label)

                    if label == "person" and conf > 0.5:
                        motion_detected = True  # Consider as motion

        if motion_detected:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            image_filename = f"{timestamp}.jpg"
            image_path = os.path.join(DETECTED_IMAGES_DIR, image_filename)
            cv2.imwrite(image_path, frame)

            send_alert("Motion Detected", 1.0, image_path)

        # Show Video Feed
        cv2.imshow("Motion, Object & Face Detection", frame)
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

