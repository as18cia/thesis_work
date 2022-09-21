import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preparation.fetch_abstracts import MetadataService


class PapersToAbstracts:
    """
    This class fetches abstracts for papers, using the Meta service written by Omar Arab Oghli.
    """

    def __init__(self):
        self.abstract_fetcher = MetadataService()

    def get_abstracts_for_papers(self):
        df = pd.read_csv("../data/processed/ResearchField_to_papers_to_contribution_statements.csv")
        failed_ids = set()
        df["PaperAbstract"] = np.nan

        for i, data in tqdm(df.iterrows(), total=len(df)):
            if pd.isna(data["PaperAbstract"]):
                try:
                    df.at[i, "PaperAbstract"] = self.abstract_fetcher.by_title(data["PaperTitle"])
                except:
                    failed_ids.add(data["PaperId"])

        df.to_csv("../data/processed/ResearchField_to_papers_to_contribution_statements_with_abstract.csv", index=False)
        df_failed = pd.DataFrame(list(failed_ids), columns=["Id"])
        df_failed.to_csv("../data/processed/failed_abstracts.csv",
                         index=False)
