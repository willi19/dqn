import environment
import deep_q_learner as dq

env = environment.ENV()
player = dq.qnlearner(env, [80, 20])

a = 1

player.learn(100000)

b = 1