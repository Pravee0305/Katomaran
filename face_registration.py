import cv2
import face_recognition
import sqlite3
from datetime import datetime
import numpy as np
import os

class FaceRegistrationSystem:
    def __init__(self):
        # Initialize database
        self.init_database()
        
        # Initialize webcam
        self.video_capture = cv2.VideoCapture(0)
        
        # Create directory for storing face images if it doesn't exist
        self.image_dir = "face_images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def init_database(self):
        # Connect to SQLite database (creates if not exists)
        self.conn = sqlite3.connect('face_registry.db')
        self.cursor = self.conn.cursor()
        
        # Create table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                image_path TEXT NOT NULL,
                registration_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def capture_face(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            
            # Find all face locations in the frame
            face_locations = face_recognition.face_locations(frame)
            
            # Draw rectangle around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Video', frame)
            
            # Press 'c' to capture, 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(face_locations) > 0:
                return frame, face_locations[0]  # Return first face found
            elif key == ord('q'):
                return None, None

    def register_face(self):
        print("Press 'c' to capture face when ready, or 'q' to quit")
        frame, face_location = self.capture_face()
        
        if frame is None:
            return False
        
        # Get name for the face
        name = input("Enter name for the captured face: ")
        
        # Get face encoding
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.image_dir, f"{name}_{timestamp}.jpg")
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        cv2.imwrite(image_path, face_image)
        
        # Store in database
        self.cursor.execute(
            "INSERT INTO faces (name, face_encoding, image_path) VALUES (?, ?, ?)",
            (name, face_encoding.tobytes(), image_path)
        )
        self.conn.commit()
        
        print(f"Successfully registered {name}")
        return True

    def list_registered_faces(self):
        self.cursor.execute("SELECT name, registration_timestamp FROM faces")
        faces = self.cursor.fetchall()
        
        if not faces:
            print("No faces registered yet.")
            return
        
        print("\nRegistered Faces:")
        print("Name | Registration Time")
        print("-" * 40)
        for name, timestamp in faces:
            print(f"{name} | {timestamp}")

    def cleanup(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.conn.close()

def main():
    system = FaceRegistrationSystem()
    
    while True:
        print("\nFace Registration System")
        print("1. Register new face")
        print("2. List registered faces")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            system.register_face()
        elif choice == '2':
            system.list_registered_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    
    system.cleanup()

if __name__ == "__main__":
    main() 