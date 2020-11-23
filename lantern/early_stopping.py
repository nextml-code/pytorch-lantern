from typing import Optional

from lantern import FunctionalBase


class EarlyStopping(FunctionalBase):
    best_score: Optional[float] = None
    scores_since_improvement: int = -1

    # def __init__(self, best_score=None, scores_since_improvement=-1):
    #     super().__init__(
    #         best_score=best_score,
    #         scores_since_improvement=scores_since_improvement,
    #     )

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
