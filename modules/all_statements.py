import pandas as pd

from modules.orkg_client import OrkgClient
from path_creator import path


class Statements:
    def __init__(self):
        self.client = OrkgClient()

    def save_all_statements(self):
        all_statements = self.client.get_all_statements()
        df = pd.DataFrame(all_statements, columns=["id", "subject", "predicate", "object"])
        df.to_csv(path("all_statements_raw.csv"), index=False)


if __name__ == '__main__':
    s = Statements()
    s.save_all_statements()
