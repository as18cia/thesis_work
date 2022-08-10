import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preparation.fetch_abstracts import MetadataService


class PapersToAbstracts:

    def __init__(self):
        self.abstract_fetcher = MetadataService()

    def get_abstracts_for_papers(self):
        df = pd.read_csv("../data/processed/ResearchFields_to_Papers_flattened.csv")
        failed_ids = set()
        df["PaperAbstract"] = np.nan

        for i, data in tqdm(df.iterrows(), total=len(df)):
            if pd.isna(data["PaperAbstract"]):
                try:
                    df.at[i, "PaperAbstract"] = self.abstract_fetcher.by_title(data["PaperTitle"])
                except:
                    failed_ids.add(data["PaperId"])
            if i == 50:
                break

        df.to_csv("../data/processed/re_field_to_paper_to_abstract.csv.csv", index=False)
        df_failed = pd.DataFrame(list(failed_ids), columns=["Id"])
        df_failed.to_csv("../data/processed/abstract_failed.csv", index=False)
