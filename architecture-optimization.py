import random
from functools import cached_property, partial
from typing import Self, Sequence
from deap import base, algorithms, creator, tools
from sklearn import datasets, metrics, model_selection, neural_network
from tqdm.contrib.concurrent import process_map
import numpy as np

import warnings

MIN_LENGTH = 0
MAX_LENGTH = 8
MIN_NEURONS = 2
MAX_NEURONS = 8
RANDOM_STATE = 42


class Individual(list):
    def __init__(self, values: Sequence | None = None) -> Self:
        if values is None:
            size = random.randint(MIN_LENGTH, MAX_LENGTH)
            return super().__init__(random.randint(MIN_NEURONS, MAX_NEURONS) for _ in range(size))
        return super().__init__(values)

    @cached_property
    def model(self):
        return neural_network.MLPClassifier(hidden_layer_sizes=self, max_iter=300, random_state=RANDOM_STATE)

    def evaluate(self, x_train, x_test, y_train, y_test):
        with warnings.catch_warnings(action="ignore"):
            ypred = self.model.fit(x_train, y_train).predict(x_test)
        error = metrics.mean_squared_error(y_test, ypred)
        size_error = sum(self)
        return error, size_error

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", Individual, fitness=creator.FitnessMin)

def main(ngen: int, population: int, halloffame: int = 5):
    print('Carregando Dataset')
    # X, y = datasets.make_classification(5000)
    X, y = datasets.load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    evaluate = partial(Individual.evaluate, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    print('Construindo Toolbox')
    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxMessyOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=MIN_NEURONS, up=MAX_NEURONS, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate)
    toolbox.register("map", process_map, max_workers=6, chunksize=8, ncols=80)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(halloffame)
    stats = tools.Statistics()
    stats.register('min', lambda pop: min(ind.fitness.values for ind in pop))

    print('Iniciando Algoritmo')
    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.2, ngen=ngen, halloffame=hof, stats=stats)
    print(log)
    for ind in hof:
        error, size = ind.fitness.values
        print('Error: %s, Size: %s' % (error, size))
    print('Best model hidden layers: %s - error: %d' % (hof[0], hof[0].fitness.values[0]))

if __name__ == '__main__':
    main(ngen=48, population=24)