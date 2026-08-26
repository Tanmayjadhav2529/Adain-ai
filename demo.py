import torch


def main():
    x = torch.tensor([1, 2])
    print(x)
    print("CUDA is available: ", torch.cuda.is_available())


if __name__ == "__main__":
    main()