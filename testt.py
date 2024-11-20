import pygame

# Initialize the mixer
pygame.mixer.init()

# Load the music file
pygame.mixer.music.load("music.wav")

# Play the music
pygame.mixer.music.play()

# Keep the program running to allow the music to play
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
