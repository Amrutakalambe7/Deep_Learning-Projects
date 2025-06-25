#Importing necessary libraries
import cv2
import pygame
import random
import numpy as np
from hand_detector import HandDetector
from hand_gesture import get_hand_gesture

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Rock Paper Scissors")

# Load images
rock_img = pygame.image.load("assets/rock.jpeg")
paper_img = pygame.image.load("assets/paper.jpeg")
scissors_img = pygame.image.load("assets/scissor.jpeg")
font = pygame.font.Font(None, 36)

# Game variables
user_score = 0
comp_score = 0
choices = ["rock", "paper", "scissors"]
detector = HandDetector()

def get_image(choice):
    return {"rock": rock_img, "paper": paper_img, "scissors": scissors_img}[choice]

def get_result(user, comp):
    if user == comp:
        return "Draw"
    elif (user == "rock" and comp == "scissors") or \
         (user == "scissors" and comp == "paper") or \
         (user == "paper" and comp == "rock"):
        return "You Win"
    else:
        return "Computer Wins"

# Initialize webcam and video writer
cap = cv2.VideoCapture(0)
clock = pygame.time.Clock()
fps = 10  # recording speed

# Define VideoWriter for recording
output_size = (640 * 2, 480)  # Width doubled to fit both screens
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("game_record.avi", fourcc, fps, output_size)

running = True
while running:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    # Hand detection
    landmarks = detector.detect_hand(frame)
    user_choice = get_hand_gesture(landmarks)
    comp_choice = random.choice(choices) if user_choice else None

    result = ""
    if user_choice and comp_choice:
        result = get_result(user_choice, comp_choice)
        if result == "You Win":
            user_score += 1
        elif result == "Computer Wins":
            comp_score += 1

    # Display webcam
    cv2.imshow("Webcam", frame)

    # Render Pygame screen
    screen.fill((255, 255, 255))
    if user_choice:
        screen.blit(get_image(user_choice), (100, 100))
    if comp_choice:
        screen.blit(get_image(comp_choice), (400, 100))
    score_text = font.render(f"You: {user_score}  Comp: {comp_score}", True, (0, 0, 0))
    result_text = font.render(f"Result: {result}", True, (255, 0, 0))
    screen.blit(score_text, (200, 20))
    screen.blit(result_text, (200, 60))
    pygame.display.flip()

    # Capture Pygame surface as image
    pg_surface = pygame.display.get_surface()
    pg_data = pygame.surfarray.array3d(pg_surface)
    pg_frame = np.transpose(pg_data, (1, 0, 2))  # Convert from (width, height, channel) to (height, width, channel)
    pg_frame = cv2.cvtColor(pg_frame, cv2.COLOR_RGB2BGR)

    # Resize OpenCV and Pygame frames to same height
    frame_resized = cv2.resize(frame, (640, 480))
    pg_frame_resized = cv2.resize(pg_frame, (640, 480))

    # Combine side-by-side
    combined_frame = np.hstack((frame_resized, pg_frame_resized))
    video_writer.write(combined_frame)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(fps)

# Release resources
cap.release()
cv2.destroyAllWindows()
video_writer.release()
pygame.quit()
