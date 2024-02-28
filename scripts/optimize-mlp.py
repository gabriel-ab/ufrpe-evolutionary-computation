# %%
from pathlib import Path
import dataclasses as dc
import random
import time
from typing import Self
import warnings
from functools import cached_property, partial
import json

import pandas as pd
from sklearn import metrics
import spacy
import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from tqdm.contrib.concurrent import process_map

# %%
MULTIPROCESS = True
NUM_GENERATIONS = 16
POPULATION_SIZE = 32
RANDOM_STATE = 42

DATA_BASE_DIR = Path("../data/")
CLASSIFICATION_REPORT_OUTPUT = DATA_BASE_DIR / "report-mlp.json"

if MULTIPROCESS:
    from tqdm.contrib.concurrent import process_map
    def map_function(func, iterable):
        return process_map(func, iterable, max_workers=6, chunksize=8, ncols=80)
else:
    from tqdm import tqdm
    def map_function(func, iterable):
        return map(func, tqdm(iterable))

# %%
def float_field(start: float, end: float):
    def factory():
        return start + random.random() * (end - start)

    return dc.field(default_factory=factory)


def int_list_field(minval: int, maxval: int, minsize: int, maxsize: int) -> list[int]:
    def factory() -> list[int]:
        return [
            random.randint(minval, maxval)
            for _ in range(random.randint(minsize, maxsize))
        ]

    return dc.field(default_factory=factory)


def int_field(start: int, end: int) -> int:
    return dc.field(default_factory=partial(random.randint, start, end))


def str_field(*options: str) -> str:
    return dc.field(default_factory=partial(random.choice, options))


def bool_field() -> bool:

    def factory() -> bool:
        return bool(random.randint(0, 1))

    return dc.field(default_factory=factory)


@dc.dataclass
class Individual:
    activation: str = str_field("relu", "logistic", "tanh")
    solver: str = str_field("lbfgs", "sgd", "adam")
    alpha: float = float_field(0.00001, 0.1)
    learning_rate_init: float = float_field(0.01, 0.1)
    power_t: float = float_field(0.3, 0.7)
    momentum: float = float_field(0.8, 0.99)
    beta_1: float = float_field(0.8, 0.99)
    beta_2: float = float_field(0.99, 0.999)
    epsilon: float = float_field(5e-9, 5e-8)

    @cached_property
    def model(self):
        return MLPClassifier((8, 8), **dc.asdict(self), random_state=RANDOM_STATE)

    def mutate(self) -> tuple[Self]:
        "Take one field and re-generate i'ts value"
        field = random.choice(dc.fields(self))
        setattr(self, field.name, field.default_factory())
        if hasattr(self, 'model'):
            del self.model
        return (self,)

    def mate_onepoint(self, other) -> tuple[Self, Self]:
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

    def mate_oneattr(self, other) -> tuple[Self, Self]:
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

    def evaluate(self, x_train, x_test, y_train, y_test) -> tuple[float]:
        with warnings.catch_warnings(action="ignore"):
            ypred = self.model.fit(x_train, y_train).predict(x_test)

        return (
            metrics.precision_score(y_test, ypred, average='micro'),
            metrics.recall_score(y_test, ypred, average='micro'),
        )

# %%
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", Individual, fitness=creator.FitnessMin)

# %%
def preprocess():
    X_train_encoded_path = DATA_BASE_DIR / "X_train_encoded.npy"
    X_test_encoded_path = DATA_BASE_DIR / "X_test_encoded.npy"

    if X_train_encoded_path.exists():
        X_train_encoded = np.load(X_train_encoded_path)
        X_test_encoded = np.load(X_test_encoded_path)
    else:
        nlp = spacy.load("en_core_web_lg")
        X_train = pd.read_csv(DATA_BASE_DIR / "X_train.csv", index_col=0)
        X_test = pd.read_csv(DATA_BASE_DIR / "X_test.csv", index_col=0)
        X_train_encoded = np.array([nlp(x["input"]).vector for _, x in X_train.iterrows()], dtype=np.float32)
        X_test_encoded = np.array([nlp(x["input"]).vector for _, x in X_test.iterrows()], dtype=np.float32)
        np.save(X_train_encoded_path, X_train_encoded)
        np.save(X_test_encoded_path, X_test_encoded)

    y_train = pd.read_csv(DATA_BASE_DIR / "y_train.csv", index_col=0)['human_evaluation']
    y_test = pd.read_csv(DATA_BASE_DIR / "y_test.csv", index_col=0)['human_evaluation']

    return y_train, y_test, X_train_encoded, X_test_encoded


def run(ngen: int, population: int, halloffame: int = 5):
    print("Carregando Dataset")

    y_train, y_test, X_train_encoded, X_test_encoded = preprocess()

    evaluate = partial(
        Individual.evaluate,
        x_train=X_train_encoded,
        x_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )

    print("Construindo Toolbox")
    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", Individual.mate_oneattr)
    toolbox.register("mutate", Individual.mutate)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate)
    toolbox.register("map", map_function)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(halloffame)
    stats = tools.Statistics()
    stats.register("min", lambda pop: min(ind.fitness.values for ind in pop))
    stats.register("max", lambda pop: max(ind.fitness.values for ind in pop))

    print("Iniciando Algoritmo")
    elapsed_time = time.perf_counter()
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, ngen=ngen, halloffame=hof, stats=stats)
    elapsed_time = time.perf_counter() - elapsed_time

    print(log)
    for i, ind in enumerate(hof, 1):
        p,r = ind.fitness.values
        print(f"hof {i} - Precision: {p} - Recall {r} - Individual: {ind}")

    model = hof[0].model
    report = {
        'report': classification_report(y_test, model.fit(X_train_encoded, y_train).predict(X_test_encoded), output_dict=True),
        'best_params': dc.asdict(hof[0]),
        'elapsed_time': elapsed_time,
        'logbook': log
    }
    return report


def test():
    y_train, y_test, X_train_encoded, X_test_encoded = preprocess()
    print('X shape', X_train_encoded.shape, 'y shape:', y_train.shape)

    evaluate = partial(
        Individual.evaluate,
        x_train=X_train_encoded,
        x_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )
    x = Individual()
    print(x)
    print(evaluate(x))
    print(classification_report(y_test, x.model.predict(X_test_encoded)))

# %%
if __name__ == "__main__":
    report = run(ngen=NUM_GENERATIONS, population=POPULATION_SIZE)
    json.dump(report, CLASSIFICATION_REPORT_OUTPUT.open('w'), indent=2)
    print(report)
