# python3
# -*- coding: utf-8 -*-
# @File    : UnionHypothesisSpace.py
# @Desc    : Book 1-2
# @Project : ML-Algorithm
# @Time    : 9/10/19 6:05 PM
# @Author  : Loopy
# @Contact : peter@mail.loopy.tech
# @License : CC BY-NC-SA 4.0 (subject to project license)

from .DatasetSpace import DatasetSpace
import json
import pandas as pd
import matplotlib.pyplot as plt


class UnionHypothesisSpace:
    """
    Figure out how the number of hypothsis change with the disjunction become longer
    """

    def __init__(self, dataset=None):
        if dataset is None:
            return
        self.__data = dataset
        self.__space = DatasetSpace(dataset)

        self.__features = list(self.__data.columns)
        self.__features.remove("target")

        self.sample_space = self.get_sample_space()

        # history hypothsis_code
        self.his_hypothsis_code = dict()
        self.his_hypothsis_code["k=1"] = self.get_conj_hypothesis()
        for i, h in enumerate(self.his_hypothsis_code["k=1"]):
            self.his_hypothsis_code["k=1"][i] = self.encode(h)

        # shown hypothsis_codes
        self.hypothsis_code_pool = [False] * 2 ** (len(self.sample_space))

    def get_conj_hypothesis(self):
        return self.__space.get_hypothesis_space()

    def get_sample_space(self):
        return self.__space.get_sample_space()

    def encode(self, hypothesis):
        """
        Encode a hypothsis to a unique code
        :param hypothesis: list
        :return: int the unique code
        """
        code = ""
        for s in self.sample_space:
            flag = "1"
            for i in range(len(self.__features)):
                if s[i] != hypothesis[i] and hypothesis[i] != "*":
                    flag = "0"
            code += flag
        return int(code, 2)

    def union(self, k):
        """
        Try union the hypothsis and skip redundant ones
        :param k:
        :return:
        """
        self.his_hypothsis_code["k=" + str(k + 1)] = []
        for i, h_l in enumerate(self.his_hypothsis_code["k=" + str(k)]):
            for j, h_r in enumerate(self.his_hypothsis_code["k=1"]):
                if not self.hypothsis_code_pool[(h_l | h_r)]:
                    self.hypothsis_code_pool[(h_l | h_r)] = True
                    self.his_hypothsis_code["k=" + str(k + 1)].append((h_l | h_r))
                print("k={} i={} j={}".format(k, i, j), end="\r")

    def run(self, k=1):
        """
        runner
        :param k: int the k to start
        :return: None
        """
        max_hypothsis_number = 2 ** (len(self.sample_space))
        count = 0
        while count < max_hypothsis_number:
            self.union(k)
            self.save_report(k)
            k += 1

            count = 0
            for h in self.hypothsis_code_pool:
                if h:
                    count += 1
        self.plot(k)

    def load(self):
        """
        load result from res.json
        :return:
        """
        with open("../temp/res.json", "r") as f:
            res = json.loads(f.read())

        self.__data = res["dataset"]
        self.__space = DatasetSpace(res["dataset"])
        self.his_hypothsis_code = res["his_hypothsis_code"]
        self.hypothsis_code_pool = res["hypothsis_code_pool"]

        self.sample_space = self.get_sample_space()

        self.__features = list(self.__data.columns)
        self.__features.remove("target")

    def save_report(self, k):
        """
        save result to res.json and report
        :param k: int current k
        :return:  None
        """
        with open("../temp/res.json", "w") as f:
            res = str(
                {
                    "dataset": self.__data,
                    "his_hypothsis_code": self.his_hypothsis_code,
                    "hypothsis_code_pool": self.hypothsis_code_pool,
                }
            )

            f.write(res)
        count = 0
        for h in self.hypothsis_code_pool:
            if h:
                count += 1
        print(
            "{}\n"
            "Working on {}-term-disjunction:\n"
            "Number of {}-term-disjunction hypothsis:{}\n"
            "Number of hypothsis:{}/{} = {}%".format(
                "=" * 79,
                k + 1,
                k,
                len(self.his_hypothsis_code["k=" + str(k)]),
                count,
                2 ** (len(self.sample_space)),
                round(count / 2 ** (len(self.sample_space)) * 100, 2),
            )
        )

    def plot(self, k):
        """
        plot k-number of hypothsis
        :param k: int max number of disjunction term
        :return: None
        """
        x = list(range(k))
        y = [len(self.his_hypothsis_code["k=" + str(i + 1)]) for i in range(k)]
        plt.plot(x, y)
        plt.show()
        plt.savefig("../temp/1.png")


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

    space = UnionHypothesisSpace(melon_data)
    space.run()

    # when you have to continue
    # space = UnionHypothesisSpace()
    # space.load()
    # space.run(10)
