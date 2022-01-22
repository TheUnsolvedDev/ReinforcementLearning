import gym

env = gym.make('LunarLander-v2')
episodes = 1000

for episode in range(1,episodes + 1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        score += reward
        
        
    
    print('Episode: {}\tScore: {}'.format(episode, score))