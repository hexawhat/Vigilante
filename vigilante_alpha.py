import cv2
import time
import datetime
import os

# DIR
path_Recordings = 'Data/Recordings/'
path_NewFaces = 'Data/NewFaces/'

# CAPTURE DEVICE
cap = cv2.VideoCapture(0)#('rtsp://admin:admin@192.168.0.31/iphone/11?admin:admin&')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # Load classifier

# CONFIG - FACES
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # VIDEO FORMAT---> Save as MP4

faces_sensitivity = 5
faces_scaleFactor = 1.3
bodies_sensitivity = 5  # Face neighbors: The lower the number the more face(neighbors) are detected (5 seems ok)
bodies_scaleFactor = 1.3  # The lower the number the best accuracy but the slower it is, min 1 (1.3 works best)

# CONFIG - VIDEO FILE SETTINGS
frame_rate = 20

# CONFIG - FEATURES - MODS
new_face_finder_mode = True


def find_new_faces():
    new_faces = face_cascade.detectMultiScale(gray, faces_scaleFactor,
                                              faces_sensitivity)  # variable needed for def find_new_faces():
    for (x, y, w, h) in new_faces:  # Goes trough each parameter of rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0),
                      5)  # Draw rectangle with parameter on faces variable
        roi_gray = gray[y:y + w,
                   x:x + w]  # RegionOfIntereset Get location of face start at y and end on w, start on x and end on w
        roi_color = frame[y:y + h, x:x + w]  # do the same but in color image
        eyes = eye_cascade.detectMultiScale(roi_gray, faces_scaleFactor,
                                            faces_sensitivity)  # Now Look for any of the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                          5)  # Draw rectangle on color ROI
        cv2.imwrite(filename=path_NewFaces + 'IMG ' + current_time + '.jpg', img=roi_gray)
    return new_faces, roi_gray


while True:
    _, frame = cap.read()
    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print('Debug: ' + gray)
    faces = face_cascade.detectMultiScale(gray, faces_scaleFactor, faces_sensitivity)
    bodies = face_cascade.detectMultiScale(gray, bodies_scaleFactor, bodies_sensitivity)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            out = cv2.VideoWriter(f"{path_Recordings} + Vid_{current_time}.mp4", fourcc, frame_rate, frame_size)
            print("Started Recording!\nPress \'q\' to 💀 the recording")
            if new_face_finder_mode:
                find_new_faces()
    elif detection:
        if not timer_started:
            timer_started = True
            detection_stopped_time = time.time()
        else:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')

    if detection:
        out.write(frame)

    # debug:
    # for (x, y, width, height) in faces:
    #    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

# CONFIG - DISPLAY
    #cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
