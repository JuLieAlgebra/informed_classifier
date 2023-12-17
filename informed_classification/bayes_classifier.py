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
        return c.p(x)

    def joint(self, c: GenerativeModel, x: np.array) -> float:
        """Computes the posterior numerator (before normalization)"""
        return self.likelihood(c, x) * self.prior[c]


if __name__ == "__main__":
    d = 100
    nominal = NominalModel(d)
    disrupted = DisruptedModel(d)
    b = BayesClassifier([0.5, 0.5], [nominal, disrupted])

    sample = disrupted.sample(1)
    posterior = b.posterior(np.zeros(nominal.dim))
    assert np.isclose(np.sum(posterior), 1.0)
    print(posterior)

    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(sample[0])), sample[0])
    plt.show()
