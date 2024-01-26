import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Game
import pygad
import numpy as np
import tensorflow as tf
import keras
import math
import random
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

USER_IN = "replay"
REPLAY_FOLDER = "Replays/"
FOLDER_NAME = "full"
HEADER_FILE = REPLAY_FOLDER + FOLDER_NAME + "/header.txt"
REPLAY_GEN = 47
REPLAY_POP = 3
FRAME_RATE = 60
CURSOR_COUNT = 4
RECURRENT_NODE_COUNT = 1
MAX_TICKS = 350
POP_SIZE = 250
GEN_NUM = 100
ELITE_NUM = math.ceil(POP_SIZE/100)

IN_SIZE = 1 + 2 + CURSOR_COUNT * 3 + RECURRENT_NODE_COUNT
OUT_SIZE = 5 + RECURRENT_NODE_COUNT
HIDDEN_SIZE = math.ceil((IN_SIZE + OUT_SIZE) / 2)
TOT_PARAM_NUM = IN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUT_SIZE + HIDDEN_SIZE + OUT_SIZE


def createNeuralNetwork():
    nn = keras.Sequential([
        keras.layers.Input(shape=(IN_SIZE,)),
        keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(OUT_SIZE, activation=tf.nn.tanh)
    ])
    return nn

def loadModel(filepath):
    with open(filepath, "rb") as file:
        solution = np.load(file)
    return solutionToNetwork(solution)

def calcModel(model, inp, rec_node_values):
    i = np.append(inp, rec_node_values)
    i = i.reshape(1, -1)
    i = tf.convert_to_tensor(i)
    return model.predict(i, verbose = 0)[0]

def solutionToNetwork(solution):
    weights = np.split(solution, np.cumsum([IN_SIZE*HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE*OUT_SIZE]))
    weights[0] = weights[0].reshape(IN_SIZE, HIDDEN_SIZE)
    weights[1] = weights[1].reshape(HIDDEN_SIZE,)
    weights[2] = weights[2].reshape(HIDDEN_SIZE, OUT_SIZE)
    weights[3] = weights[3].reshape(OUT_SIZE,)
    nn = createNeuralNetwork()
    nn.set_weights(weights)
    return nn

def saveSolution(solution, replay):
    top_folder = REPLAY_FOLDER + FOLDER_NAME
    gen_num = len(os.listdir(top_folder)) - 2
    gen_folder = top_folder + "/gen_" + str(gen_num)
    while True:
        try:
            pop_num = len(os.listdir(gen_folder))
            pop_folder = gen_folder + "/pop_" + str(pop_num)
            os.mkdir(pop_folder)
            break
        except FileExistsError:
            continue
    with open(pop_folder + "/solution.npy", "wb") as file:
        np.save(file, solution)
    with open(pop_folder + "/replay.pkl", "wb") as file:
        pickle.dump(replay, file)

def playReplay(gen, pop):
    folder = REPLAY_FOLDER + FOLDER_NAME + "/gen_" + str(gen) + "/pop_" + str(pop)
    with open(folder + "/replay.pkl", "rb") as file:
        replay = pickle.load(file)
        print(replay.fitness)
        replay.play(FRAME_RATE)

def getFitness(ga_instance, solution, solution_index):
    app = Game.App()
    nn = solutionToNetwork(solution)
    tick = 0
    rec_node_values = [0 for _ in range(RECURRENT_NODE_COUNT)]
    while not app.player.isDead and tick < MAX_TICKS:
        i = np.array(Game.getOutput(app, CURSOR_COUNT))
        i = np.append(i, rec_node_values)
        i = i.reshape(1, -1)
        i = tf.convert_to_tensor(i)
        o = nn.predict(i, verbose = 0)[0]
        for _ in range(5):
            Game.handleInput(app, o)
            Game.onStep(app)
        rec_node_values = o[OUT_SIZE-RECURRENT_NODE_COUNT:]
        tick += 1
    app.replay.fitness = app.fitness
    saveSolution(solution, app.replay)
    return app.fitness

def updateGenerationLoadingBar(cur_gen, max_gen):
    gen_progress = cur_gen / max_gen
    bar_length = 30
    gen_bar = "-" * int(gen_progress * bar_length)
    gen_empty_space = " " * (bar_length - int(gen_progress * bar_length))
    print(f"generation: {cur_gen-1} - [{gen_bar}{gen_empty_space}]", end="", flush=True)

def createInitialPopulation(pop_size):
    pass

def onGeneration(ga_instance):
    cur_gen = ga_instance.generations_completed
    last_fitness = ga_instance.last_generation_fitness
    if cur_gen != GEN_NUM:
        while True:
            try:
                os.mkdir(REPLAY_FOLDER + FOLDER_NAME + "/gen_" + str(cur_gen))
                break
            except FileExistsError:
                continue
    mean_fit = sum(last_fitness) / len(last_fitness)
    best_fit = max(last_fitness)
    simularity = np.mean(cosine_similarity(ga_instance.population))
    updateGenerationLoadingBar(cur_gen, GEN_NUM)
    print(f" __ average : {mean_fit} , best : {best_fit} , similarity : {simularity}")
    with open(HEADER_FILE, "a") as file:
        file.write(f"generation {cur_gen-1}: average = {round(mean_fit)} , best = {round(best_fit)} , similarity : {simularity}\n")

def getSortedGen(gen):
    folder = REPLAY_FOLDER + FOLDER_NAME + "/gen_" + str(gen)
    replays = []
    solutions = []
    pop_files = os.listdir(folder)
    for pop in pop_files:
        with open(folder + "/" + pop + "/replay.pkl", "rb") as file:
            replays.append(pickle.load(file))
        with open(folder + "/" + pop + "/solution.npy", "rb") as file:
            solutions.append(np.load(file))
    inds = [i for i in range(len(replays))]
        
    def sortKey(a):
        return a[0].fitness

    zipped_array = sorted(list(zip(replays, solutions, inds)), key=sortKey, reverse=True)
    replays, solutions, inds = zip(*zipped_array)
    return replays, solutions, inds

def playGenInOrder(gen):
    folder = REPLAY_FOLDER + FOLDER_NAME + "/gen_" + str(gen)
    replays = []
    pop_files = os.listdir(folder)
    for pop in pop_files:
        with open(folder + "/" + pop + "/replay.pkl", "rb") as file:
            r = pickle.load(file)
            replays.append(r)
    
    def sortKey(a):
        return a.fitness

    replays.sort(key=sortKey, reverse=True)

    for replay in replays:
        print(replay.fitness)
        replay.play(FRAME_RATE)

def playFullGame():
    replays, solutions, inds = getSortedGen(REPLAY_GEN)
    network = solutionToNetwork(solutions[0])
    print(f"pop num: {inds[0]}")
    print(f"fitness: {replays[0].fitness}")
    app = Game.App()
    rec_node_values = [0 for _ in range(RECURRENT_NODE_COUNT)]
    while not app.player.isDead:
        i = np.array(Game.getOutput(app, CURSOR_COUNT))
        i = np.append(i, rec_node_values)
        i = i.reshape(1, -1)
        i = tf.convert_to_tensor(i)
        o = network.predict(i, verbose = 0)[0]
        for _ in range(5):
            Game.handleInput(app, o)
            Game.onStep(app)
        rec_node_values = o[OUT_SIZE-RECURRENT_NODE_COUNT:]
    app.replay.fitness = app.fitness
    print(app.points)
    app.replay.play(FRAME_RATE)
    return app.replay

def train():
    os.mkdir(REPLAY_FOLDER + FOLDER_NAME)
    os.mkdir(REPLAY_FOLDER + FOLDER_NAME + "/gen_0")
    with open(HEADER_FILE, "w") as file:
        file.write(f"{FOLDER_NAME}\n")
        file.write(f"cursor count: {CURSOR_COUNT}\n")
        file.write(f"recurrent node count: {RECURRENT_NODE_COUNT}\n")
        file.write(f"population size: {POP_SIZE}\n")
        file.write(f"max ticks: {MAX_TICKS}\n")
        file.write(f"elite num: {ELITE_NUM}\n")
    ga_instance = pygad.GA(
        num_generations=GEN_NUM,
        num_parents_mating=math.ceil(POP_SIZE/3),
        fitness_func=getFitness,
        # fitness_batch_size=,
        # initial_population=createInitialPopulation(pop_size),
        sol_per_pop=POP_SIZE,
        num_genes=TOT_PARAM_NUM,
        init_range_low=-1,
        init_range_high=1,
        gene_type=float,
        #parent_selection_type="rws",
        #keep_parents=0,
        keep_elitism=ELITE_NUM,
        # K_tournament=,
        crossover_type="two_points",
        # crossover_probability=,
        mutation_type="random",
        # mutation_probability=,
        # mutation_by_replacement=,
        mutation_percent_genes=6,
        # mutation_num_genes=,
        random_mutation_min_val=-0.1,
        random_mutation_max_val=0.1,
        # gene_space=,
        # allow_duplicate_genes=,
        # on_start=onStart,
        # on_fitness=onFitness,
        # on_parents=,
        # on_crossover=,
        # on_mutation=,
        on_generation=onGeneration,
        # on_stop=,
        # delay_after_gen=,
        # save_best_solutions=,
        # save_solutions=,
        # suppress_warnings=,
        # stop_criteria=,
        parallel_processing=["process", 8],
        # random_seed=,
        # logger=pygad.logging.Logger("logger")
    )

    ga_instance.run()

def genPopFullGame():
    model_path = REPLAY_FOLDER + FOLDER_NAME + "/gen_" + str(REPLAY_GEN) + "/pop_" + str(REPLAY_POP) + "/solution.npy"
    with open(model_path, "rb") as file:
        network = np.load(file)
    network = solutionToNetwork(network)
    replay = playFullGame(network)
    print(replay.fitness)
    replay.play(FRAME_RATE)

def main():
    if USER_IN == "train":
        train()
    elif USER_IN == "replay":
        playFullGame()
    print("exited")

if __name__ == "__main__":
    main()