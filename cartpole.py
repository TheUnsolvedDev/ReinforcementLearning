import gym

env = gym.make('CartPole-v0')
episodes = 10

for i in range(episodes):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        score += reward
        
    print('Episode: {}\tScore: {}'.format(i, score))
