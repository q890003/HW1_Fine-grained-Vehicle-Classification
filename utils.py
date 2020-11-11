import torch


def get_gpu():
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU ...")
    else:
        print("CUDA is not available. Using CPU ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


if __name__ == '__main__':
     print(get_gpu())
