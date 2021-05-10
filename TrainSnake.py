from SnakeGameAI import Snake, location
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(11, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self):
        torch.save(self.state_dict(), 'trainedSnake.pth')

class SnakeNetTarget:
    def __init__(self, model, gamma):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr= 0.001)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

# Kinda Changed
    def trainStep(self, snakeInfo, nextSnakeInfo, action, reward, done):
        snakeInfo = torch.tensor(snakeInfo, dtype = torch.float)
        if (len(snakeInfo.shape) == 1):
            snakeInfo = torch.unsqueeze(snakeInfo, 0)
            nextSnakeInfo = torch.tensor(nextSnakeInfo, dtype = torch.float)
            nextSnakeInfo = torch.unsqueeze(nextSnakeInfo, 0)
            action = torch.tensor(action, dtype = torch.long)
            action = torch.unsqueeze(action, 0)
            reward = torch.tensor(reward, dtype = torch.float)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        else:
            nextSnakeInfo = torch.tensor(nextSnakeInfo, dtype = torch.float)
            action = torch.tensor(action, dtype = torch.long)
            reward = torch.tensor(reward, dtype = torch.float)
        pred = self.model(snakeInfo)
        target = pred.clone()
        for i in range(len(done)):
            QNew = reward[i]
            if not done[i]:
                QNew = reward[i] + self.gamma * torch.max(self.model(nextSnakeInfo[i]))
            target[i][torch.argmax(action[i]).item()] = QNew
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class SnakeTrainer:
    def __init__(self):
        self.numOfGames = 0
        self.epsilon = 80
        self.gamma = .9
        self.model = SnakeNet()
        self.trainer = SnakeNetTarget(self.model, gamma = self.gamma)
        self.memory = deque(maxlen = 100000)

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

    def getAction(self, snakeInfo):
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

    def learnTrainStep(self, snakeInfo, nextSnakeInfo, action, reward, done):
        self.trainer.trainStep(snakeInfo, nextSnakeInfo, action, reward, done)
        self.memory.append((snakeInfo, nextSnakeInfo, action, reward, done))

    def trainGame(self):
        if (len(self.memory) > 1000):
            sample = random.sample(self.memory, 1000)
        else:
            sample = self.memory
        snakeInfo, nextSnakeInfo, actions, rewards, dones = zip(*sample)
        self.trainer.trainStep(snakeInfo, nextSnakeInfo, actions, rewards, dones)

if __name__ == '__main__':
    snake = Snake(1280, 960)
    snakeTrainer = SnakeTrainer()
    highScore = 0
    while (snakeTrainer.numOfGames != 500):
        snakeInfo = snakeTrainer.getSnakeInfo(snake)
        action = snakeTrainer.getAction(snakeInfo)
        endGame, score, reward = snake.AIControl(action)
        nextSnakeInfo = snakeTrainer.getSnakeInfo(snake)
        snakeTrainer.learnTrainStep(snakeInfo, nextSnakeInfo, action, reward, endGame)
        if endGame:
            snake.reset_game()
            snakeTrainer.numOfGames += 1
            snakeTrainer.trainGame()
            print('Game', snakeTrainer.numOfGames, 'Score', score, 'Current High Score:', highScore)
            if score > highScore:
                highScore = score
                snakeTrainer.model.save()