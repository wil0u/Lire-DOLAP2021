import torch


class Config:
    __instance = None
    device_ = None
    precision_ = torch.float32

    @staticmethod
    def device():
        return Config.getInstance().device_

    @staticmethod
    def precision():
        return Config.getInstance().precision_

    @staticmethod
    def set_device_gpu():
        Config.getInstance().device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_device_cpu():
        Config.getInstance().device_ = torch.device("cpu")

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Config.__instance is None:
            Config()
        return Config.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Config.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Config.__instance = self
