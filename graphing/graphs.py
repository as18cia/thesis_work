import pandas as pd
import matplotlib.pyplot as plt


class Graphs:

    @staticmethod
    def value_graph():
        df = pd.read_csv(r"C:\Users\mhrou\Desktop\Orkg\ResearchFields_to_Contributions.csv")
        contributions1, contributions2 = list(df["#OfContributions"][:260]), list(df["#OfContributions"][260:])
        # contributions1 = contributions1[1:]
        plt.bar([i for i in range(len(contributions1))], contributions1)
        plt.yticks([0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000])
        plt.show()

        # plt.bar([259 + i for i in range(len(contributions2))], contributions2)
        # plt.yticks([0, 1, 2, 3, 4, 5])
        # plt.show()

        # path = r"C:\Users\mhrou\Desktop\Orkg\ResearchFields_to_Contributions.csv" + str(i) + ".png"


if __name__ == '__main__':
    g = Graphs()
    g.value_graph()
