from SnakeGameAI import Snake, location
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), 'trainedSnake.pth')


class SnakeTest:
    def __init__(self):
        self.game_number = 0
        self.model = SnakeNet()
        self.model.load_state_dict(torch.load('trainedSnake.pth'))
        self.epsilon = 80

    def getSnakeInfo(self, snake):
        # Gives direction on where snake is going
        going = [False, False, False, False]
        if snake.AIDirection == 1:
            going[0] = True
        if snake.AIDirection == 2:
            going[1] = True
        if snake.AIDirection == 3:
            going[2] = True
        if snake.AIDirection == 4:
            going[3] = True
        # Food Positions
        foodXY = [snake.food.x < snake.head.x, snake.food.x > snake.head.x, snake.food.y < snake.head.y, snake.food.y > snake.head.y]
        # Snake Positions
        positions = [location(snake.snake[0].x - 40, snake.snake[0].y), location(snake.snake[0].x - 40, snake.snake[0].y), location(snake.snake[0].x, snake.snake[0].y - 40), location(snake.snake[0].x, snake.snake[0].y + 40)]
        # Checks Collision with potential positions and going direction
        checkLeftLeft = snake.checkCollision(positions[0]) and going[0]
        checkRightRight = snake.checkCollision(positions[1]) and going[1]
        checkUpUp = snake.checkCollision(positions[2]) and going[2]
        checkDownDown = snake.checkCollision(positions[3]) and going[3]
        check1 = checkLeftLeft or checkRightRight or checkUpUp or checkDownDown
        checkLeftUp = snake.checkCollision(positions[0]) and going[2]
        checkRightDown = snake.checkCollision(positions[1]) and going[3]
        checkUpRight = snake.checkCollision(positions[2]) and going[1]
        checkDownLeft = snake.checkCollision(positions[3]) and going[0]
        check2 = checkLeftUp or checkRightDown or checkUpRight or checkDownLeft
        checkLeftDown = snake.checkCollision(positions[0]) and going[3]
        checkRightUp = snake.checkCollision(positions[1]) and going[2]
        checkUpLeft = snake.checkCollision(positions[2]) and going[0]
        checkDownRight = snake.checkCollision(positions[3]) and going[1]
        check3 = checkLeftDown or checkRightUp or checkUpLeft or checkDownRight
        snakeInfo = [foodXY[0], foodXY[1], foodXY[2], foodXY[3], check1, check2, check3, going[0], going[1], going[2], going[3]]
        return snakeInfo

    def get_action(self, snakeInfo):
        self.epsilon -= self.numOfGames
        temp = random.randint(0, 200)
        if (temp < self.epsilon):
            predictedDirection = random.randint(0, 2)
        else:
            log = torch.tensor(snakeInfo, dtype = torch.float)
            prediction = self.model(log)
            predictedDirection = torch.argmax(prediction).item()
        move = [None, None, None]
        for x in range(3):
            if (move[x] == None):
                move[x] = 0
            if (predictedDirection == x):
                move[x] = 1
        self.epsilon = 80
        return move

if __name__ == '__main__':
    snake = Snake(1280, 960)
    snakeTest = SnakeTest()
    highScore = 0
    while snakeTest.game_number != 500:
        logistics = snakeTest.getSnakeInfo(snake)
        action = snakeTest.get_action(logistics)
        end_game, score, reward = snake.AIControl(action)
        if end_game:
            snake.reset_game()
            snakeTest.game_number += 1
            if score > record:
                record = score
            print('Game', snakeTest.game_number, 'Score', score, 'High Score:', highScore)
