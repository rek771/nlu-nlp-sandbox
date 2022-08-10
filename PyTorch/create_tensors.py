import torch
import numpy as np


def describe(x):
    """
    Во-первых, опишем вспомогательную функцию, describe(x), для вывода различных характеристик тензора x, например типа тензора, его размерности и содержимого
    :param x:
    :return:
    """
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


# тензор со случайными значениями
describe(torch.Tensor(2, 3))

# случайное равномерное распределение
describe(torch.rand(2, 3))

# случайное нормальное распределение
describe(torch.randn(2, 3))

# заполняем нулями
describe(torch.zeros(2, 3))

# заполняем единицами
x = torch.ones(2, 3)
describe(x)

# перезаполняем "на месте"(без пересоздания) пятерками
x.fill_(5)
describe(x)

# заполняем листом
x = torch.Tensor([[1, 2, 3],
                  [4, 5, 6]])
describe(x)

# заполняем слуайной матрицей numpy
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))
