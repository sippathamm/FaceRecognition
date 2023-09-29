import face_recognition
import cv2
import pickle
import serial

# port = "/dev/cu.usbserial-210"
# arduino = serial.Serial(port, 9600)

model = pickle.load(open("model.svm", "rb"))
print("[INFO] Loaded model successfully")

this_frame = True

name = {0: "Kanapat_Netthaisongh",
        1: "Pongpisit_Kambongkan",
        2: "Sippawit_Thammawiset"}

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    if not _:
        break

    if this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        detected_face = face_recognition.face_locations(small_frame, model="hog")

        for i in range(len(detected_face)):
            top, right, bottom, left = detected_face[i]
            encoded_face = face_recognition.face_encodings(small_frame, detected_face)[i]

            predict_name_index = model.predict([encoded_face])[0]
            predict_probability = model.predict_proba([encoded_face])[0]
            confidence = round(predict_probability[predict_name_index], 2)

            unknown = True

            if confidence > 0.90:
                unknown = False

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if not unknown:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 10)
                cv2.putText(frame, str(name[predict_name_index]), (left, top - 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, str(confidence),
                            (right, top - 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

                # arduino.write(b"F")
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 10)
                cv2.putText(frame, "Unknown", (left, top - 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

                # arduino.write(b"S")

    this_frame = not this_frame

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
