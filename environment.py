import random
import numpy as np

class ENV:

    def __init__(self, size=4):
        self.board = np.zeros((size, size))
        self.size = size
        self.generate()
        self.generate()
        self.action_size = 4
        self.score = 0
        self.update_possible_action()
        self.state_size = size*size

    def init(self):
        self.board = np.zeros((self.size,self.size))
        self.generate()
        self.generate()
        self.update_possible_action()
        self.score = 0

    def rotate(self, direction):
        ret = np.copy(self.board)
        for _ in range(direction):
            ret = np.rot90(ret)
        return ret

    def move(self, direction):
        if direction not in self.possible_action:
            return -1
        reward = 0
        self.board = self.rotate(direction)
        for col in range(self.size):
            nx = 0
            last_number = -1
            for row in range(self.size):
                val = self.board[row][col]
                if val == 0:
                    continue
                if last_number == val:
                    self.board[nx][col] += 1
                    reward += self.board[nx][col]
                    nx += 1
                    last_number = -1
                    continue
                else:
                    if last_number != -1:
                        nx += 1
                    last_number = self.board[nx][col] = val
            if last_number != -1:
                nx += 1
            for row in range(nx, self.size):
                self.board[row][col] = 0
        self.board = self.rotate((4 - direction) % 4)
        self.generate()
        self.score += reward
        self.update_possible_action()
        return reward

    @staticmethod
    def generate_random_number():
        if random.random() > 0.1:
            return 1
        return 2

    def generate(self):
        cand = np.where(self.board == 0)
        if len(cand) is 0:
            return
        ind = random.randrange(len(cand[0]))
        self.board[cand[0][ind]][cand[1][ind]] = self.generate_random_number()

    def show(self):
        print("score: "+str(self.score))
        print(self.board)

    def movable(self, direction):
        tmp_board = self.rotate(direction)

        for col in range(self.size):
            for row in range(self.size-1):
                if tmp_board[row][col] == 0 and tmp_board[row+1][col] != 0:
                    return True
                elif tmp_board[row][col] != 0 and tmp_board[row][col] == tmp_board[row+1][col]:
                    return True
        return False

    def update_possible_action(self):
        self.possible_action = []
        for direction in range(4):
            if self.movable(direction):
                self.possible_action.append(direction)

    def get_state(self):
        ret = self.board.reshape(self.state_size).copy()
        return ret

    def terminate(self):
        return len(self.possible_action) == 0

if __name__ == "__main__":
    env = ENV()
    while not env.terminate():
        env.show()
        move = int(input())
        env.move(move)
    env.show()