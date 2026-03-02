import cv2
import mediapipe as mp
import csv

label = input("Enter gesture label: ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()

        # Check if frame captured
        if not success or frame is None:
            print("Camera frame not captured")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []

                # Extract 21 landmark coordinates
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                    row.append(lm.z)

                # Add gesture label
                row.append(label)

                writer.writerow(row)

        cv2.imshow("Dataset Collection", frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()