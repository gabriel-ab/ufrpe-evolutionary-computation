import dataclasses as dc
import random
import warnings
from functools import cached_property, partial
from typing import Literal, Self, Sequence

import numpy as np
from deap import algorithms, base, creator, tools
from sklearn import datasets, metrics, model_selection, neural_network
from tqdm.contrib.concurrent import process_map

MIN_LENGTH = 0
MAX_LENGTH = 8
MIN_NEURONS = 2
MAX_NEURONS = 8
RANDOM_STATE = 42


def float_field(start: float, end: float):
    def factory():
        return start + random.random() * (end - start)

    return dc.field(default_factory=factory)


def list_field() -> list[int]:
    def factory() -> list[int]:
        return [
            random.randint(MIN_NEURONS, MAX_NEURONS)
            for _ in range(random.randint(MIN_LENGTH, MAX_LENGTH))
        ]

    return dc.field(default_factory=factory)


def int_field(start: float, end: float):
    return dc.field(default_factory=partial(random.randint, start, end))


def str_field(*options: str) -> str:
    return dc.field(default_factory=partial(random.choice, options))


@dc.dataclass(slots=True)
class Individual:
    hidden_layer_sizes: list = list_field()
    activation: str = str_field("relu", "identity", "logistic", "tanh")
    solver: str = str_field("lbfgs", "sgd", "adam")
    alpha: float = float_field(0.0001, 0.1)
    learning_rate: str = str_field("constant", "invscaling", "adaptive")
    learning_rate_init: float = float_field(0.001, 0.1)
    power_t: float = float_field(0.2, 0.8)
    momentum: float = float_field(0.2, 0.9)
    beta_1: float = float_field(0.7, 0.99)
    beta_2: float = float_field(0.99, 0.9999)
    epsilon: float = float_field(1e-9, 1e-7)

    @cached_property
    def model(self):
        return neural_network.MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=300,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            momentum=self.momentum,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )

    def mutate(self, indpb: float = 0.2):
        "Take one field and re-generate i'ts value"
        fields = dc.fields(self)
        if random.random() < indpb:
            field = random.choice(fields)
            setattr(self, field.name, field.default_factory())
        return self

    def mate_onepoint(self, other):
        field_names = [field.name for field in dc.fields(self)]
        point = random.randint(0, len(field_names))
        child1 = type(self)(
            **{
                attr: getattr(self if i < point else other, attr)
                for i, attr in enumerate(field_names)
            }
        )
        child2 = type(self)(
            **{
                attr: getattr(other if i < point else self, attr)
                for i, attr in enumerate(field_names)
            }
        )
        return child1, child2

    def mate_oneattr(self, other):
        field_names = [field.name for field in dc.fields(self)]
        point = random.randint(0, len(field_names))
        child1 = type(self)(
            **{
                attr: getattr(self if i == point else other, attr)
                for i, attr in enumerate(field_names)
            }
        )
        child2 = type(self)(
            **{
                attr: getattr(other if i == point else self, attr)
                for i, attr in enumerate(field_names)
            }
        )
        return child1, child2

    def evaluate(self, x_train, x_test, y_train, y_test):
        with warnings.catch_warnings(action="ignore"):
            ypred = self.model.fit(x_train, y_train).predict(x_test)
        error = metrics.mean_squared_error(y_test, ypred)
        size_error = sum(self)
        return error, size_error


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", Individual, fitness=creator.FitnessMin)


def main(ngen: int, population: int, halloffame: int = 5):
    print("Carregando Dataset")
    # X, y = datasets.make_classification(5000)
    X, y = datasets.load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2
    )

    evaluate = partial(
        Individual.evaluate,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("Construindo Toolbox")
    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxMessyOnePoint)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=MIN_NEURONS, up=MAX_NEURONS, indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate)
    toolbox.register("map", process_map, max_workers=6, chunksize=8, ncols=80)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(halloffame)
    stats = tools.Statistics()
    stats.register("min", lambda pop: min(ind.fitness.values for ind in pop))

    print("Iniciando Algoritmo")
    pop, log = algorithms.eaSimple(
        pop, toolbox, 0.2, 0.2, ngen=ngen, halloffame=hof, stats=stats
    )
    print(log)
    for ind in hof:
        error, size = ind.fitness.values
        print("Error: %s, Size: %s" % (error, size))
    print(
        "Best model hidden layers: %s - error: %d" % (hof[0], hof[0].fitness.values[0])
    )


if __name__ == "__main__":
    main(ngen=48, population=24)
