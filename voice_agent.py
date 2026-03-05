import pyttsx3
import time
import os

engine = pyttsx3.init()

last_spoken = ""

print("AI Voice Agent Started...")

while True:

    if os.path.exists("gesture_output.txt"):

        with open("gesture_output.txt", "r") as f:
            gesture = f.read().strip()

        if gesture != "" and gesture != last_spoken:

            print("Speaking:", gesture)

            engine.say(gesture)
            engine.runAndWait()

            last_spoken = gesture

    time.sleep(0.5)