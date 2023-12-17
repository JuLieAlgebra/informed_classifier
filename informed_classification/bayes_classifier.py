import numpy as np

from informed_classification.generative_models import (
    DisruptedModel,
    GenerativeModel,
    NominalModel,
)


class BayesClassifier:
    """Gives MAP and posterior"""

    def __init__(self, prior: list[float], classes: list[GenerativeModel]):
        # prior is p(c)
        assert np.isclose(np.sum(prior), 1.0)
        self.prior = dict(zip(classes, prior))
        self.classes = classes

    def classify(self, x: np.array) -> np.array:
        """Based on data point, which model is more likely? Returns the MAP"""
        MAP = np.argmax(self.posterior(x), axis=0)
        return MAP  # self.classes[MAP]

    def posterior(self, x: np.array) -> list[float]:
        """Computes the posteriors for each class"""
        evidence = self.evidence(x)
        return np.array([self.joint(c, x) / evidence for c in self.classes])

    def evidence(self, x: np.array) -> float:
        """Computes the denominator of Bayes rule, p(x)"""
        return np.sum([self.joint(c, x) for c in self.classes], axis=0)

    def likelihood(self, c: GenerativeModel, x: np.array) -> float:
        """Computes the likelihood of the data point, x, given model c"""
        return np.float128(c.p(x))

    def joint(self, c: GenerativeModel, x: np.array) -> float:
        """Computes the posterior numerator (before normalization)"""
        return self.likelihood(c, x) * self.prior[c]


if __name__ == "__main__":
    # np.random.seed(0)
    d = 100

    nominal = NominalModel(d)
    disrupted = DisruptedModel(d)

    prior = [0.5, 0.5]
    bayes = BayesClassifier(prior, [nominal, disrupted])

    n = 10000
    nominal_data = nominal.sample(n)
    disrupted_data = disrupted.sample(n)

    correct_nominal = 0
    for x in nominal_data:
        if bayes.classify(x) == nominal:
            correct_nominal += 1

    correct_disrupted = 0
    for x in disrupted_data:
        if bayes.classify(x) == disrupted:
            correct_disrupted += 1

    print("Nominal Accuracy:", correct_nominal / n)
    print("Disrupt Accuracy:", correct_disrupted / n)
    print("Mean's Posterior:", bayes.posterior(disrupted.dist.mean))
