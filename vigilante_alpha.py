import cv2
import time
from datetime import datetime
import os
# Attandance
import numpy as np
import face_recognition

# DIR
path_Recordings = 'Data/Recordings/'
path_NewFaces = 'Data/NewFaces/'
path_knownFaces ='Data/KnownFaces'

# Attendance
faces_knownFaces = []
faces_names = []
faces_myList = os.listdir(path_knownFaces)
print(f'Debug: List of known faces:\n {faces_myList}')


#   Scans folder to create class names and reference
for cl in faces_myList:
    curImg = cv2.imread(f'{path_knownFaces}/{cl}')
    faces_knownFaces.append(curImg)
    faces_names.append(os.path.splitext(cl)[0])

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


def markAttendance(name):  # name
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        print('DEBUG: myDataList', myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            print(nameList)
        #if name not in nameList:  # comment this line out if you do not want to check
            # if the name is already on the list
            now = datetime.now()
            dtString = now.strftime('%d-%m-%Y-%H-%M-%S')
            f.writelines(f'\n{name},{dtString}')


def findEncodings(faces_knownFaces):  #
    encodeList = []
    for img in faces_knownFaces:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(faces_knownFaces)

while True:
    _, frame = cap.read()
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, faces_scaleFactor, faces_sensitivity)
    bodies = face_cascade.detectMultiScale(gray, bodies_scaleFactor, bodies_sensitivity)
    if len(faces) + len(bodies) > 0:
        if new_face_finder_mode:
            find_new_faces()
        if detection:
            timer_started = False
        else:
            detection = True
            out = cv2.VideoWriter(f"{path_Recordings} + Vid_{current_time}.mp4", fourcc, frame_rate, frame_size)
            print("Started Recording!\nPress \'q\' to STOP the RECORDING")
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
        # Face Recon
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)  # resize to make it faster
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # convert into RGB
        #  Step 1 Detect face location
        facesCurFrame = face_recognition.face_locations(imgS)  # Find face location on current frame small image
        print("facesCurFrame",facesCurFrame)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # face encoding for c f
        #   Step 3
        #   Compare the results
        #   loop trough all faces
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # zip makes possible to grab face and encoding
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # compare faces
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # find distance
            matchIndex = np.argmin(faceDis)  # Create index to identify the person
            #  Uses Index to display
            if matches[matchIndex]: # compare
                name = faces_names[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
               # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw rectangle / optional
               # cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED) # draw rectangle optional
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # draw name / optional
                markAttendance(name)

    # debug:
    # for (x, y, width, height) in faces:
    #    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

# CONFIG - DISPLAY
    cv2.imshow("Vigilante soft. Beta Version", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
