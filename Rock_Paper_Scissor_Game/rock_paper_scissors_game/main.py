import cv2
import pygame
import random
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

cap = cv2.VideoCapture(0)
clock = pygame.time.Clock()

running = True
while running:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
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

    # Display camera
    cv2.imshow("Webcam", frame)
    screen.fill((255, 255, 255))

    # Pygame display
    if user_choice:
        screen.blit(get_image(user_choice), (100, 100))
    if comp_choice:
        screen.blit(get_image(comp_choice), (400, 100))

    score_text = font.render(f"You: {user_score}  Comp: {comp_score}", True, (0, 0, 0))
    result_text = font.render(f"Result: {result}", True, (255, 0, 0))
    screen.blit(score_text, (200, 20))
    screen.blit(result_text, (200, 60))

    pygame.display.flip()
    clock.tick(1)  # Limit to 1 frame per second

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
