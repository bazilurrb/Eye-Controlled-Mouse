import cv2                              # OpenCV for image and video processing
import mediapipe as mp                 # MediaPipe for face and eye landmark detection
import pyautogui                       # PyAutoGUI for controlling the mouse

cam = cv2.VideoCapture(0)             # Open the default webcam (camera index 0)

# Initialize MediaPipe FaceMesh with refined landmarks (includes eyes, iris, etc.)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen width and height for mapping eye position to screen coordinates
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()             # Read a frame from the webcam
    frame = cv2.flip(frame, 1)        # Flip the frame horizontally for mirror view

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
    output = face_mesh.process(rgb_frame)               # Process the frame to detect face landmarks
    landmark_points = output.multi_face_landmarks       # Get the list of detected landmarks

    frame_h, frame_w, _ = frame.shape                   # Get frame dimensions

    if landmark_points:                                 # If landmarks are detected
        landmarks = landmark_points[0].landmark         # Use the first face detected

        # Loop through iris landmarks (474 to 477 are near the right eye's iris)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)               # Convert normalized x to pixel x
            y = int(landmark.y * frame_h)               # Convert normalized y to pixel y
            cv2.circle(frame, (x, y), 3, (0, 255, 0),)   # Draw a small green circle at the point

            if id == 1:                                 # Use landmark 475 (approx. iris center) for cursor movement
                screen_x = screen_w / frame_w * x       # Map x from frame to screen
                screen_y = screen_h / frame_h * y       # Map y from frame to screen
                pyautogui.moveTo(screen_x, screen_y)    # Move the mouse to the calculated position

        # Get two landmarks around the left eye to detect blinking
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)               # Convert normalized x to pixel x
            y = int(landmark.y * frame_h)               # Convert normalized y to pixel y
            cv2.circle(frame, (x, y), 3, (0, 0, 255),)   # Draw red dots on left eye landmarks

        # If the vertical distance between two landmarks is small => blink detected
        if (left[0].y - left[1].y) < 0.01:
            pyautogui.click()                           # Perform a mouse click
            pyautogui.sleep(1)                          # Sleep for 1 second to prevent multiple clicks
            print("click")                              # Log the click action

    cv2.imshow("Eye Controlled Mouse", frame)           # Show the frame in a window
    cv2.waitKey(1)                              # Wait 1ms between frames (needed for imshow)
