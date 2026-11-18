"""A real selector extension, not a mock reconstruction backend."""

from reconlab.errors import ValidationError
from reconlab.interfaces import ViewSelector


class EveryNthTrainingView(ViewSelector):
    def __init__(self, stride: int = 2):
        if type(stride) is not int or stride < 1:
            raise ValidationError("stride must be a positive integer")
        self.stride = stride

    def select(self, views):
        chosen = {view.id for view in [v for v in views if v.split == "train"][:: self.stride]}
        return tuple(v for v in views if v.split != "train" or v.id in chosen)


def register(registry):
    registry.register(ViewSelector, "every_nth", EveryNthTrainingView)
