from torch import device, cuda


class DefaultDeviceResolver:
    def resolve(self) -> device:
        if cuda.is_available():
            return device("cuda")
        return device("cpu")
