import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Haar cascades for Eye Detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Load the Haar cascades for facial expression recognition
# expression_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_smile.xml')


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    # Start the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        # Iterate over the faces
        for (x, y, w, h) in faces:
            # Crop the face region
            roi_gray = gray[y:y+h, x:x+w]

            faces = face_cascade.detectMultiScale(frame_gray)
            for (x, y, w, h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.ellipse(frame, center, (w//2, h//2),
                                    0, 0, 360, (255, 0, 255), 4)
                faceROI = frame_gray[y:y+h, x:x+w]
                # -- In each face, detect eyes
                eyes = eye_cascade.detectMultiScale(faceROI)
                for (x2, y2, w2, h2) in eyes:
                    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                    radius = int(round((w2 + h2)*0.25))
                    frame = cv2.circle(frame, eye_center,
                                       radius, (255, 0, 0), 4)
            cv2.imshow('Capture - Face detection', frame)

            # Detect facial expressions in the face region
        #     expressions = expression_cascade.detectMultiScale(roi_gray, 1.3, 5)

        #     # Iterate over the expressions
        #     for (ex, ey, ew, eh) in expressions:
        #         # Draw a rectangle around the expression
        #         cv2.rectangle(frame, (x+ex, y+ey),
        #                       (x+ex+ew, y+ey+eh), (255, 255, 0), 2)

        # Encode the frame as a JPEG image
        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        # Yield the frame to the client
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
