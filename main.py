import environment
import deep_q_learner as dq

env = environment.ENV()
player = dq.qnlearner(env, [80, 20])

player.learn(100000)