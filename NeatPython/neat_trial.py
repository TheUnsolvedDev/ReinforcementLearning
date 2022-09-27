from neat import config, population, genome, reproduction, species, stagnation, nn, reporting, statistics, checkpoint
import neat
import numpy as np
import gym

render = True
render2 = True

highest_scores = []


def eval_genomes(genomes, configure):
    global highest_scores

    member = 0
    high_score = 0
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 100000
    for genome_id, genome in genomes:
        print('Member:', member)
        score = 0
        observation = env.reset()
        observation = observation[0]
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, configure)

        while not done:
            inputs = [observation[0], observation[1],
                      observation[2], observation[3]]
            if render:
                env.render()
            action = round(net.activate(inputs)[0])
            observation, reward, done, info,_ = env.step(action)
            if done:
                break
            else:
                score += 1

        genome.fitness = score
        if score > high_score:
            high_score = score

        member += 1
        print('Fitness:', score)
        print('Highest_Scores_Over_Generation:', highest_scores)
        print('------------------------------')
    env.close()
    highest_scores.append(high_score)


def render_winner(net):
    print('Rendering Winner')

    env = gym.make('CartPole-v1',render_mode = 'human')
    observation = env.reset()
    observation = observation[0]
    done = False

    while not done:
        action = round(net.activate(
            [observation[0], observation[1], observation[2], observation[3]])[0])
        env.render()
        observation, reward, done, info,_ = env.step(action)
        if done:
            break
    env.close()
    print('---------------------------------')


if __name__ == '__main__':
    configure = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                     neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                     'config-feedforward')

    p = neat.population.Population(configure)
    p.add_reporter(neat.reporting.StdOutReporter(True))
    stats = neat.statistics.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, configure)
    if render2:
        render_winner(winner_net)
    input("Enter to Quit: ")
