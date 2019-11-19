import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_melon(return_array=False):
    # pd.read_csv('https://raw.githubusercontent.com/CQU-AI/Watermelon-book-puzzles/master/Chapter-04/melon_data.csv',index_col=0)
    melon_data = pd.DataFrame(
        [
            ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.46, "是"],
            ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, "是"],
            ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, "是"],
            ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, "是"],
            ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, "是"],
            ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, "是"],
            ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, "是"],
            ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, "是"],
            ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, "否"],
            ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, "否"],
            ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, "否"],
            ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, "否"],
            ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"],
            ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"],
            ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.36, 0.37, "否"],
            ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"],
            ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否"],
        ],
        columns=["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"],
    )

    if return_array:
        for f in ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"]:
            melon_data[f] = LabelEncoder().fit_transform(melon_data[f])
        return melon_data.values
    else:
        return melon_data
