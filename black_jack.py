"""
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.
    
    ### Description
    
    Card Values:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.
    
    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.
    
    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.
    
    ### Action Space
    
    There are two actions: stick (0), and hit (1).
    
    ### Observation Space
    
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).
    
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).
    
    ### Rewards
    
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
        +1.5 (if <a href="#nat">natural</a> is True)
        +1 (if <a href="#nat">natural</a> is False)
        
    ### Arguments
    
    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```
    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
    
    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).
    
    ### Version History
    * v0: Initial versions release (1.0.0)
"""
import gym
import numpy as np
import time

env = gym.make('Blackjack-v1')
env = env.unwrapped

n_actions = env.action_space.n
n_observations = env.observation_space

print(n_actions, n_observations)
print((env.reset()))

print(env.observation_space.sample())


if __name__ == '__main__':
    for i in range(100):
        print(env.reset())
        done = False
        while not done:
            # env.render()
            state,reward,done,info = env.step(env.action_space.sample())
            # time.sleep(0.1)
        print(i)


    env.close()
