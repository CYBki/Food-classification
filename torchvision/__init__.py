# Minimal stub of torchvision package

class transforms:
    class ToTensor:
        def __call__(self, x):
            return x


class io:
    @staticmethod
    def read_image(path):
        return []
