import Game
import tensorflow as tf
import keras
import numpy as np
import random
import math
import concurrent.futures
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

CURSOR_COUNT = 4
RECURRENT_NODE_COUNT = 0
IN_SIZE = 1 + 2 + CURSOR_COUNT * 3 + RECURRENT_NODE_COUNT
OUT_SIZE = 5 + RECURRENT_NODE_COUNT

REPLAY_FOLDER = "Replays/"
SAVED_MODELS_FOLDER = "SavedModels/"

def createNetwork():
    model = keras.Sequential([
        keras.layers.Input(shape=(IN_SIZE,)),
        #keras.layers.Dense(units=(IN_SIZE+OUT_SIZE)/2, activation='relu'),
        keras.layers.Dense(OUT_SIZE, activation=tf.nn.tanh)
        #keras.layers.Dense(OUT_SIZE, activation=tf.nn.tanh, bias_initializer=keras.initializers.glorot_normal(), kernel_initializer=keras.initializers.glorot_normal())
        #keras.layers.Dense(OUT_SIZE, activation=tf.nn.tanh, kernel_initializer='zeros', bias_initializer='zeros')
    ])
    #model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def getSimularity(models):
    # model_parameters = [np.concatenate([w.flatten() for w in model.get_weights()]) for model in models]
    # #print(model_parameters)
    # kmeans = KMeans(n_clusters=3, n_init='auto')
    # clusters = kmeans.fit_predict(model_parameters)
    # return davies_bouldin_score(model_parameters, clusters)
    model_weights = [np.concatenate([w.flatten() for w in model.get_weights()]) for model in models]

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(model_weights)

    return np.mean(similarity_matrix)


def select_individuals(fitness_scores, pop_size):
    # print("fit scores:")
    # print(fitness_scores)
    new_fitness_scores = fitness_scores[:]
    min_fit = min(new_fitness_scores)
    if min_fit < 0:
        for i in range(len(new_fitness_scores)):
            new_fitness_scores[i] += min_fit * -1
    #new_fitness_scores = [i * i for i in new_fitness_scores]
    total_fitness = sum(new_fitness_scores)
    if total_fitness == 0: return []
    probabilities = [score / total_fitness for score in new_fitness_scores]
    # print("probs:")
    # print(probabilities)

    selected_indices = np.random.choice(len(new_fitness_scores), size=pop_size, p=probabilities)
    # print("inds:")
    # print(selected_indices)
    return selected_indices

def crossover(parent1, parent2):
    child1 = createNetwork()
    child2 = createNetwork()
    for i in range(len(child1.layers)):
        cur_weight_crossover = 0
        cur_bias_crossover = 0
        child1_data = parent1.layers[i].get_weights()
        child2_data = parent2.layers[i].get_weights()
        weight_num = 0
        for node in child1_data[0]: weight_num += len(node)
        weight_crossover_point = random.randint(0, weight_num)
        bias_num = len(child1_data[1])
        bias_crossover_point = round(bias_num * weight_crossover_point/weight_num)
        for inode in range(len(child1_data[0])):
            for iweight in range(len(child1_data[0][inode])):
                if cur_weight_crossover < weight_crossover_point:
                    child1_data[0][inode][iweight] = child2_data[0][inode][iweight]
                else:
                    child2_data[0][inode][iweight] = child1_data[0][inode][iweight]
                cur_weight_crossover += 1
        
        for inode in range(len(child1_data[1])):
            if cur_bias_crossover < bias_crossover_point:
                child1_data[1][inode] = child2_data[1][inode]
            else:
                child2_data[1][inode] = child1_data[1][inode]
            cur_bias_crossover += 1
        child1.layers[i].set_weights(child1_data)
        child2.layers[i].set_weights(child2_data)

    return child1, child2

def mutate(net, mutation_rate=0.08):
    for i in range(len(net.layers)):
        net_data = net.layers[i].get_weights()
        for inode in range(len(net_data[0])):
            for iweight in range(len(net_data[0][inode])):
                if random.random() < mutation_rate:
                    net_data[0][inode][iweight] += random.random() * 0.1 - 0.05
        for inode in range(len(net_data[1])):
            if random.random() < mutation_rate:
                net_data[1][inode] += random.random() * 0.05 - 0.025
        net.layers[i].set_weights(net_data)

def evaluateNetwork(network, max_ticks):
    app = Game.App()
    tick = 0
    rec_node_values = [0 for _ in range(RECURRENT_NODE_COUNT)]
    while not app.player.isDead and tick < max_ticks:
        i = np.array(Game.getOutput(app, CURSOR_COUNT))
        i = np.append(i, rec_node_values)
        i = i.reshape(1, -1)
        i = tf.convert_to_tensor(i)
        # print("in:")
        # print(i)
        o = network.predict(i, verbose = 0)[0]
        # print("out:")
        # print(o)
        rec_node_values = o[OUT_SIZE-RECURRENT_NODE_COUNT:OUT_SIZE]
        
        for _ in range(5):
            Game.handleInput(app, o)
            Game.onStep(app)
        

        tick+=1
    
    Game.getDistanceFitness(app)
    app.replay.fitness = app.fitness

    return app.replay

def train(pop_size, generations, name, continue_filepath = ""):
    name_num = -1
    folder_name = ""
    while True:
        try:
            folder_name = name if name_num == -1 else name + "_" + str(name_num)
            os.mkdir(REPLAY_FOLDER + folder_name)
            break
        except Exception:
            name_num += 1
    name = folder_name
    
    direct_carry_num = math.ceil(pop_size / 3)
    population = []
    if continue_filepath == "":
        for i in range(pop_size): 
            net = createNetwork()
            mutate(net)
            population.append(net)
    else:
        for i in range(pop_size):
            filepath = continue_filepath + "pop_" + str(i) + "/model.keras"
            population.append(keras.models.load_model(filepath))
    
    base_max_ticks = 200

    for gen in range(generations):
        print(f"gen {gen}:")
        gen_replays = []
        gen_models = []
        fitness_scores = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            def evaluate_network_wrapper(net):
                return evaluateNetwork(net, base_max_ticks)

            futures = [executor.submit(evaluate_network_wrapper, population[i]) for i in range(pop_size)]

            for ind, future in enumerate(concurrent.futures.as_completed(futures)):
                replay = future.result()
                model = population[ind]
                fitness_scores.append(replay.fitness)
                gen_replays.append(replay)
                gen_models.append(model)

        # for i in range(pop_size):
        #     fit, replay = evaluateNetwork(population[i], base_max_ticks)
        #     #print(f"network {i} - fitness: {fit}")
        #     fitness_scores.append(fit)
        #     gen_replays.append(replay)
        #     gen_models.append(population[i])
        
        new_population = []

        zipped_arrs = list(zip(fitness_scores, population))
        zipped_arrs = sorted(zipped_arrs, key=lambda x: x[0], reverse=True)
        sorted_fitness_scores, population = zip(*zipped_arrs)
        sorted_fitness_scores = list(sorted_fitness_scores)
        population = list(population)
        for i in range(direct_carry_num):
            new_population.append(population[i])

        indices = select_individuals(sorted_fitness_scores[0:direct_carry_num], pop_size)
        for i in range(0, len(sorted_fitness_scores), 2):
            if len(new_population) == pop_size:
                break
            parent1 = population[indices[i]]
            parent2 = population[indices[i + 1]]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        print(f"best: {round(max(fitness_scores))} - average: {round(sum(fitness_scores) / len(fitness_scores))} \
               - simularity score: {getSimularity(population)}")
        
        gen_filepath = REPLAY_FOLDER + "/" + name + "/" + "gen_" + str(gen)
        os.mkdir(gen_filepath)
        for i in range(pop_size):
            pop_filepath = gen_filepath + "/" + "pop_" + str(i)
            os.mkdir(pop_filepath)
            population[i].save(pop_filepath + "/model.keras")
            with open(pop_filepath + "/py_replay.pkl", "wb") as file:
                pickle.dump(gen_replays[i], file)

        population = new_population

def runFullGame(network):
    app = Game.App()
    rec_node_values = [0 for _ in range(RECURRENT_NODE_COUNT)]

    while not app.player.isDead:
        i = np.array(Game.getOutput(app, CURSOR_COUNT))
        i = np.append(i, rec_node_values)
        i = i.reshape(1, -1)
        i = tf.convert_to_tensor(i)
        o = network.predict(i, verbose = 0)[0]
        rec_node_values = o[OUT_SIZE-RECURRENT_NODE_COUNT:OUT_SIZE]
        
        for _ in range(5):
            Game.handleInput(app, o)
            Game.onStep(app)
    
    return app.points, app.replay

def runModel(filename):
    net = tf.keras.models.load_model(filename)
    points, replay = runFullGame(net)
    print(points)
    replay.play(60)

def main():
    while True:
        inpt = input("action:\nnew training    continue training    replay    exit\n")
        if inpt == "new training":
            train(int(input("population size: ")), int(input("number of generations: ")), \
                   input("folder name: "))
        elif inpt == "continue training":
            filepath = REPLAY_FOLDER + input("folder name: ") + "/gen_" + input("generation: ") + "/"
            train(int(input("population size: ")), int(input("number of generations: ")), \
                   input("folder name: "), filepath)
        elif inpt == "replay":
            folder = input("folder name: ")
            if folder == "back": continue
            while True:
                gen = input("generation: ")
                if gen == "back": break
                while True:
                    pop_ind = input("population index (or list): ")
                    if pop_ind == "back": break
                    elif pop_ind == "list":
                        gen_filepath = replay_filepath = REPLAY_FOLDER + folder + "/gen_" + gen
                        try:
                            for pop_folder in os.listdir(gen_filepath):
                                replay_filepath = gen_filepath + "/" + pop_folder + "/py_replay.pkl"
                                with open(replay_filepath, "rb") as file:
                                    replay : Game.Replay = pickle.load(file)
                                print(pop_folder + " - " + str(round(replay.fitness)))
                            continue
                        except Exception as e:
                            print(e)
                            print("failed - filepath: " + gen_filepath)
                            continue
                    while True:
                        replay_action = input("replay action:\nplay    run full game    back\n")
                        if replay_action == "play":
                            replay_filepath = REPLAY_FOLDER + folder + "/gen_" + gen + "/pop_" + pop_ind + "/py_replay.pkl"
                            try:
                                with open(replay_filepath, "rb") as file:
                                    replay : Game.Replay = pickle.load(file)
                                replay.play(60)
                            except Exception as e:
                                print(e)
                                print("failed - filepath: " + replay_filepath)
                                continue
                        elif replay_action == "run full game":
                            model_filepath = REPLAY_FOLDER + folder + "/gen_" + gen + "/pop_" + pop_ind + "/model.keras"
                            try:
                                model :keras.Model = keras.models.load_model(model_filepath)
                                points, full_replay = runFullGame(model)
                                print("points scored: " + str(points))
                                full_replay.play(60)
                                while True:
                                    model_action = input("save model? (Y/N) ")
                                    if model_action == "Y":
                                        model.save(SAVED_MODELS_FOLDER + input("model name: "))
                                    if model_action == "N":
                                        break
                            except Exception as e:
                                print(e)
                                print("failed - filepath: " + model_filepath)
                                continue
                        elif replay_action == "back":
                            break

                
        elif inpt == "exit":
            break

    
    print("ended")

if __name__ == "__main__":
    main()