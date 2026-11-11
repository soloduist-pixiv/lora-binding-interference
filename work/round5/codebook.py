"""Follow-up with a finite, learnable codebook and held-out prompt contexts."""
from itertools import product
import pilot

OLD = [0, 61, 73, 17, 29, 41, 53, 89, 67, 19, 31, 43, 59, 83, 47]
NEW = [0] + OLD[8:] + OLD[1:8]
KEYS = "_ABCDEFGHJKLMNP"
CONTEXTS = (
    "This request comes from desk {}.",
    "The request number is {}.",
    "For ticket {}, answer the question below.",
)


def problem(task, a, b, variant):
    if variant == "counterfactual":
        a += 1
    key = KEYS[a]
    context = CONTEXTS[b % len(CONTEXTS)].format(400 + b)
    if task == "old":
        question = (f"Which integer corresponds to key {key} in the Cedar register?"
                    if variant == "equivalent" else f"What is the code for key {key} in the Cedar register?")
        value = OLD[a]
    elif task == "related":
        question = f"What is the code for key {key} in the Maple register?"
        value = NEW[a]
    else:
        question = f"How many battery packs does robot {key} carry?"
        value = NEW[a]
    return f"{context} {question}", value


def data():
    contexts = list(range(1, 46))
    pilot.random.Random(pilot.SEED).shuffle(contexts)
    return {split: list(product(range(1, 15, 2), bs)) for split, bs in
            (("train", contexts[:22]), ("validation", contexts[22:26]), ("test", contexts[26:]))}


if __name__ == "__main__":
    pilot.STUDY = "codebook"
    pilot.OLD_STEPS = 96
    pilot.data = data
    pilot.OUT = pilot.ROOT / "outputs/round5_20261108/codebook"
    pilot.problem = problem
    pilot.OLD_TRAIN_VARIANTS = pilot.VARIANTS
    pilot.NEW_TRAIN_VARIANTS = ("original", "counterfactual")
    pilot.run()

