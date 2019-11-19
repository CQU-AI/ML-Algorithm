# python3
# -*- coding: utf-8 -*-
# @File    : DatasetSpace.py
# @Desc    : Book 1-1
# @Project : ML-Algorithm
# @Time    : 9/10/19 3:21 PM
# @Author  : Loopy
# @Contact : peter@mail.loopy.tech
# @License : CC BY-NC-SA 4.0 (subject to project license)

import pandas as pd
from tabulate import tabulate


class DatasetSpace:
    """
    Automatically generate the sample_space, hypothesis_space and version_space for a data set
    """

    def __init__(self, dataset):
        """
        :param dataset: pd.DataFrame
        """
        self.__data = dataset
        self.__features = list(self.__data.columns)
        self.__features.remove("target")

    def get_hypothesis_space(self):
        """
        :return: list (list hypothesis (unknown sample_value))
        """
        possible_values = self.get_possible_values()
        for k in possible_values.keys():
            possible_values[k] += ["*"]
        enum = Enumerator([len(possible_values[f]) for f in self.__features])
        hypothesis_space = []

        while True:
            res = [0] * len(self.__features)
            code = enum.next()
            if code == -1:
                break
            for i, e in enumerate(code):
                res[i] = possible_values[self.__features[i]][e]
            hypothesis_space.append(res)
        hypothesis_space.append(["-"] * len(self.__features))
        return hypothesis_space

    def get_sample_space(self):
        """
        :return: list (list sample (unknown sample_value))
        """
        possible_values = self.get_possible_values()
        enum = Enumerator([len(possible_values[f]) for f in self.__features])
        sample_space = []

        while True:
            res = [0] * len(self.__features)
            code = enum.next()
            if code == -1:
                break
            for i, e in enumerate(code):
                res[i] = possible_values[self.__features[i]][e]
            sample_space.append(res)
        return sample_space

    def get_version_space(self):
        """
        Check the hypothesis_space with the samples, which will get version_space
        :return:
        """
        global hypothesis_space, is_vaild
        hypothesis_space = self.get_hypothesis_space()
        is_vaild = [True] * len(hypothesis_space)

        def check(sample):
            global hypothesis_space, is_vaild
            if sample["target"]:  # positive sample
                for i, hps in enumerate(hypothesis_space):
                    for j, f in enumerate(self.__features):
                        if sample[f] != hps[j] and hps[j] != "*":
                            # hypothesis not agree with the positive sample
                            is_vaild[i] = False
                            break
            else:
                for i, hps in enumerate(hypothesis_space):
                    for j, f in enumerate(self.__features):
                        if sample[f] == hps[j]:
                            # hypothesis agree with the negative sample
                            is_vaild[i] = False
                            break
                    if set(hps) - set(["*"] * len(self.__features)) == set():
                        is_vaild[i] = False
            return sample

        self.__data.apply(check, axis=1)
        version_space = []
        for i, flag in enumerate(is_vaild):
            if flag:
                version_space.append(hypothesis_space[i])
        return version_space

    def get_possible_values(self):
        """
        Get the possible sample values
        :return: dic (key:(str feature)
                      value:(list possible_feature_values))
        """
        possible_values = {}
        for f in self.__features:
            possible_values[f] = list(self.__data[f].unique())
        return possible_values

    @property
    def hypothesis(self):
        return tabulate(
            pd.DataFrame(self.get_hypothesis_space(), columns=self.__features),
            tablefmt="pipe",
            headers="keys",
        )

    @property
    def version(self):
        return tabulate(
            pd.DataFrame(self.get_version_space(), columns=self.__features),
            tablefmt="pipe",
            headers="keys",
        )

    @property
    def sample(self):
        return tabulate(
            pd.DataFrame(self.get_sample_space(), columns=self.__features),
            tablefmt="pipe",
            headers="keys",
        )


class Enumerator:
    """
    enumerator for hypothesis_space
    """

    def __init__(self, max):
        self.code = [0] * len(max)
        self.base = max
        self.overflow = False

    def next(self):
        if self.overflow:
            return -1

        self.code[0] += 1
        if not self.carry():
            self.overflow = True
            return self.code
        else:
            return self.code

    def carry(self):
        for i, b in enumerate(self.base):
            if self.code[i] == b:
                self.code[i] = 0
                if i + 1 >= len(self.code):
                    return False
                self.code[i + 1] += 1

                self.carry()
            else:
                return True


if __name__ == "__main__":
    melon_data = pd.DataFrame(
        [
            ["青绿", "蜷缩", "浊响", True],
            ["乌黑", "蜷缩", "浊响", True],
            ["青绿", "硬挺", "清脆", False],
            ["乌黑", "稍蜷", "沉闷", False],
        ],
        columns=["色泽", "根蒂", "敲声", "target"],
    )

    space = DatasetSpace(melon_data)

    print("=" * 79 + "\n样本空间: \n" + space.sample)

    print("=" * 79 + "\n合取假设的假设空间: \n" + space.hypothesis)

    print("=" * 79 + "\n合取假设的版本空间: \n" + space.version)
