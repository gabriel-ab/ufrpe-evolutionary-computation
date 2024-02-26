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
from sklearn.ensemble import RandomForestClassifier
from tqdm.contrib.concurrent import process_map

# %%
NUM_GENERATIONS = 32
POPULATION_SIZE = 64
RANDOM_STATE = 42

DATA_BASE_DIR = Path("../data/")
CLASSIFICATION_REPORT_OUTPUT = DATA_BASE_DIR / "report-rf.json"


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
    n_estimators: int = int_field(10, 200)
    max_depth: int = int_field(1, 20)
    min_samples_split: int = int_field(2, 20)
    min_samples_leaf: int = int_field(1, 20)
    bootstrap: bool = bool_field()
    criterion: str = str_field("gini", "entropy")

    @cached_property
    def model(self):
        return RandomForestClassifier(**dc.asdict(self), random_state=RANDOM_STATE)

    def mutate(self, indpb: float = 0.2) -> tuple[Self]:
        "Take one field and re-generate i'ts value"
        fields = dc.fields(self)
        if random.random() < indpb:
            field = random.choice(fields)
            setattr(self, field.name, field.default_factory())
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

    def evaluate(self, x_train, x_test, y_train, y_test)-> tuple[float]:
        with warnings.catch_warnings(action="ignore"):
            ypred = self.model.fit(x_train, y_train).predict(x_test)
        error = metrics.mean_squared_error(y_test, ypred)
        return (error,)


# %%
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
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

    y_train = pd.read_csv(DATA_BASE_DIR / "y_train.csv", index_col=0)
    y_test = pd.read_csv(DATA_BASE_DIR / "y_test.csv", index_col=0)

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
    toolbox.register("mutate", Individual.mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate)
    toolbox.register("map", process_map, max_workers=6, chunksize=8, ncols=80)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(halloffame)
    stats = tools.Statistics()
    stats.register("min", lambda pop: min(ind.fitness.values for ind in pop))

    print("Iniciando Algoritmo")
    elapsed_time = time.perf_counter()
    pop, log = algorithms.eaSimple(
        pop, toolbox, 0.2, 0.2, ngen=ngen, halloffame=hof, stats=stats
    )
    elapsed_time = time.perf_counter() - elapsed_time

    print(log)
    for i, ind in enumerate(hof, 1):
        print(f"hof {i} - Error: {ind.fitness.values[0]} - Individual: {ind}")

    model = hof[0].model
    report = classification_report(y_test, model.predict(X_test_encoded), output_dict=True)
    report['best_params'] = dc.asdict(hof[0])
    report['elapsed_time'] = elapsed_time
    report['log_book'] = log
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
    json.dump(report, CLASSIFICATION_REPORT_OUTPUT.open('w'))
    print(report)
