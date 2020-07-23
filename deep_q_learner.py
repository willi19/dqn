import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class function_approximator(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(function_approximator, self).__init__()
        self.function = nn.Sequential(nn.Linear(inputs, hiddens[0]), nn.LeakyReLU(0.2))
        for i in range(1, len(hiddens)):
            self.function.add_module("hidden "+str(i), nn.Linear(hiddens[i-1], hiddens[i]))
            self.function.add_module("leaky relu "+str(i), nn.LeakyReLU(0.2))
        self.function.add_module("output ", nn.Linear(hiddens[-1], outputs))

    def forward(self, x):
        return self.function(x)


class qnlearner:

    def __init__(self, env, hiddens, gamma = 0.9):
        self.env = env
        self.q_function = function_approximator(env.state_size, hiddens, env.action_size)
        self.optimizer = optim.SGD(self.q_function.parameters(), lr = 0.01, momentum=0.9)
        self.gamma = gamma
        self.history = []
        self.memory = []

    def get_state(self):
        return torch.from_numpy(self.env.get_state()).float()

    def next_move(self, best_move):
        if random.random()<0.1:
            return random.randrange(self.env.action_size)
        else:
            return best_move



    def learn(self, epochs):
        for epoch in range(epochs):
            self.env.init()
            moves = 0
            self.q_function.train()
            while not self.env.terminate():
                moves += 1
                self.q_function.zero_grad()
                self.optimizer.zero_grad()
                state = self.get_state()
                q_vals = self.q_function(state)
                action = self.next_move(torch.argmax(q_vals).item())
                q_val = q_vals[action]
                reward = self.env.move(action)/10
                next_state = self.get_state()
                target_q_val = reward + self.gamma*torch.max(self.q_function(next_state).detach())
                loss = (target_q_val-q_val)**2
                loss.backward()
                self.optimizer.step()
                if moves > 1000:
                    break
            self.history.append(self.env.score)
            if epoch % 1000 == 0:
                print("epoch:  "+str(epoch)+" score: "+str(self.env.score))
                self.env.show()

    def play(self):
        self.q_function.eval()
        moves = 0
        while not self.env.terminate():
            state = self.get_state()
            action = torch.argmax(self.q_function(state)).item()
            self.env.show()
            self.env.move(action)
            if moves > 1000:
                break
