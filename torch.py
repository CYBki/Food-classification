class nn:
    """Minimal stub of torch.nn."""
    class Module:
        pass


class cuda:
    @staticmethod
    def is_available():
        return False


class device:
    def __init__(self, name):
        self.name = name


class Tensor(list):
    """Simple list-based tensor with minimal functionality."""
    def sum(self):
        return TensorSum(sum(self))


class TensorSum(int):
    def item(self):
        return int(self)


def tensor(data):
    return Tensor(data)


def eq(a, b):
    return Tensor([1 if x == y else 0 for x, y in zip(a, b)])
