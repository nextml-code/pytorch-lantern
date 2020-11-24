from typing import Optional

from lantern import FunctionalBase


class EarlyStopping(FunctionalBase):
    """Keeps track of the best score and how long ago it was calculated."""

    best_score: Optional[float] = None
    scores_since_improvement: int = -1

    def score(self, value):
        if self.best_score is None or value >= self.best_score:
            return self.replace(
                best_score=value,
                scores_since_improvement=0,
            )
        else:
            return self.replace(
                scores_since_improvement=self.scores_since_improvement + 1
            )

    def print(self):
        print(
            "".join(
                [
                    f"best score: {self.best_score}, "
                    f"scores since improvement: {self.scores_since_improvement}"
                ]
            )
        )
        return self

    def log(self):
        pass
