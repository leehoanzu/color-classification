# Import opencv for computer vision stuff
import cv2
import time

# Connect to webcam
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("start!\n")

# Wait 3 seconds to start
time.sleep(3)

# Loop through every frame until we close our webcam
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Show image
    cv2.imshow('Webcam', frame)
    
    # Checks whether q has been hit and stops the loop
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()