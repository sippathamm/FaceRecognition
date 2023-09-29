import face_recognition
import cv2


def crop_face_webcam(result_directory, target_size):
    video = cv2.VideoCapture(0)

    if not cv2.os.path.exists(result_directory):
        cv2.os.makedirs(result_directory)

    process = 1

    while True:
        _, frame = video.read()

        if not _:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        detected_face = face_recognition.face_locations(small_frame, model="hog")

        if len(detected_face) == 1:
            top, right, bottom, left = detected_face[0]
            face = small_frame[top:bottom, left:right]
            face = cv2.resize(face, target_size)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 10)

            save_path = cv2.os.path.join(result_directory, f"{process}.jpg")

            cv2.imwrite(save_path, face)

            process += 1

            print(f"[INFO] Saved {process}.jpg")

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

    print("[INFO] Completed.")


result_directory = "Webcam"
target_size = (160, 160)
crop_face_webcam(result_directory, target_size)
