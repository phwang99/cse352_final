import pygame

pygame.init()

class snake:
    def __init__(self):
        self.width = 500
        self.height = 500

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.display.fill((255, 255, 255))

if __name__ == '__main__':
    game = snake