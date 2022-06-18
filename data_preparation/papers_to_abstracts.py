import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preparation.multi_thread_caller import MultiThreadCaller
from data_preparation.orkg_client import OrkgClient
from data_preparation.re_fields_to_papers import ReFieldsToPapers
from path_creator import path, get_file_names
import patoolib
import time
import ast


class PapersToAbstracts:

    def __init__(self):
        self.client = OrkgClient()
        self.papers_to_re = ReFieldsToPapers().create_paper_to_re_mapping()

    def get_all_papers_with_doi(self):
        # todo: clean the data if this function is to be used again
        # we need paper title or doi, if both are missing then remove the paper

        papers = self.client.get_resources_by_class("Paper")
        papers_with_doi = []
        for paper in tqdm(papers, total=len(papers)):
            doi = self.client.get_doi_for_paper(paper[0])
            papers_with_doi.append((paper[0], doi, paper[1], paper[2]))
        papers_with_doi.sort(key=lambda x: len(x[2]), reverse=True)
        df = pd.DataFrame(papers_with_doi, columns=["paper_id", "paper_doi", "paper_title", "Class"])
        df.to_csv(path("all_papers_with_doi.csv"), index=False)

    def get_all_contributions(self):
        all_contributions = self.client.get_resources_by_class("Contribution")
        pd.DataFrame(all_contributions, columns=["id", "label", "classes"]).to_csv(path("all_contributions.csv"),
                                                                                   index=False)

    def get_statements_for_contributions(self):
        df = pd.read_csv(path("all_contributions.csv"))
        all_statements = []
        for i, data in tqdm(df.iterrows(), total=len(df)):
            all_statements.extend(self.client.get_statement_based_on_predicate(data["id"]))

        all_statements = [(x["id"], x["subject"], x["predicate"], x["object"]) for x in all_statements]
        df = pd.DataFrame(all_statements, columns=["id", "subject", "predicate", "object"])
        df = self._process_statements(df)
        df.to_csv(path("contribution_statements.csv"), index=False)

    def get_abstracts_for_papers(self):
        df = pd.read_csv(path("papers_to_abstracts.csv"))
        failed_ids = set()
        for i, data in tqdm(df.iterrows(), total=len(df)):
            if pd.isna(data["paper_abstract"]):
                if not pd.isna(data["paper_doi"]):
                    try:
                        pass
                        # df.at[i, "paper_abstract"] = self.metadata.by_doi(data["paper_doi"])
                    except:
                        failed_ids.add(data["paper_id"])
                        time.sleep(3)

        df.to_csv(path("papers_to_abstracts_v2.csv"), index=False)
        df_failed = pd.DataFrame(list(failed_ids), columns=["id"])
        df_failed.to_csv(path("failed_to.csv"), index=False)

    def get_papers_to_contribution(self):
        papers = pd.read_csv(path("all_papers.csv"))["Id"].tolist()
        p2c = []
        for p in tqdm(papers, total=len(papers)):
            p2c.extend(self.client.get_statement_based_on_predicate(p, "P31"))

        p2c = [(x["subject"]["id"], x["subject"]["label"], x["object"]["id"], x["object"]["label"]) for x in p2c]

        pd.DataFrame(p2c,
                     columns=["paper_id", "paper_title", "contribution_id", "contribution_label"]).to_csv(
            path("papers_to_contributions.csv"),
            index=False)

    def merge_all_data(self):
        contribution_statements = pd.read_csv(path("contribution_statements.csv"))
        papers_to_contributions = pd.read_csv(path("papers_to_contributions.csv"))
        papers_to_abstracts = pd.read_csv(path("papers_to_abstracts_v2.csv"))

        cont_statements_2_papers = contribution_statements.merge(papers_to_contributions, left_on="subject_id",
                                                                 right_on="contribution_id", how="left")
        cont_statements_to_abstracts = cont_statements_2_papers.merge(papers_to_abstracts, left_on="paper_id",
                                                                      right_on="paper_id",
                                                                      how="left")

        cont_statements_to_abstracts = cont_statements_to_abstracts[
            ["statement_id", "subject_id", "predicate_label", "object_id", "object_label", "paper_id", "paper_title_x",
             "paper_abstract"]]
        cont_statements_to_abstracts = self._is_in_abstract(cont_statements_to_abstracts)
        cont_statements_to_abstracts: pd.DataFrame = self.merged_to_research_field(cont_statements_to_abstracts)
        cont_statements_to_abstracts = self.sort_drop_reorder_for_training(cont_statements_to_abstracts)
        cont_statements_to_abstracts.to_csv(path("merged_statements_to_abstracts_v3.csv"), index=False)

    def merged_to_research_field(self, df: pd.DataFrame):
        df["research_field_id"] = np.nan
        df["research_field_label"] = np.nan
        for i, data in df.iterrows():
            if data["paper_id"] in self.papers_to_re:
                df.at[i, "research_field_id"] = self.papers_to_re[data["paper_id"]]["research_field_id"]
                df.at[i, "research_field_label"] = self.papers_to_re[data["paper_id"]]["research_field_label"]
            else:
                print("found one")
        return df

    @staticmethod
    def sort_drop_reorder_for_training(df: pd.DataFrame):
        df["helper_1"] = np.nan
        df["helper_2"] = np.nan
        for i, data in df.iterrows():
            if not pd.isna(data["research_field_id"]):
                df.at[i, "helper_1"] = int(data["research_field_id"][1:])
            if not pd.isna(data["paper_id"]):
                df.at[i, "helper_2"] = int(data["paper_id"][1:])

        df = df.sort_values(["helper_1", "helper_2"], ascending=[True, True])
        df = df.drop(columns=['helper_1', "helper_2"])
        df = df[
            ['research_field_id', 'research_field_label', 'paper_id', 'paper_title_x', 'statement_id', 'subject_id',
             'predicate_label', 'object_id', 'object_label', 'paper_abstract', 'object_in_abstract']]
        df = df[df["object_in_abstract"].isin([True])]
        return df

    @staticmethod
    def get_stats():
        statements = pd.read_csv(path("contribution_statements.csv"))
        print(statements.count())

        print("-------------- end result stats ---------------\n")
        s2ab = pd.read_csv(path("merged_statements_to_abstracts_v2.csv"))
        print(s2ab.count())

        print("-------------- out of 71479 ---------------\n")
        in_abstract = s2ab["object_in_abstract"].tolist().count(True)
        not_in_abstract = s2ab["object_in_abstract"].tolist().count(False)
        print("in_abstract: ", in_abstract)
        print("not_in_abstract: ", not_in_abstract)

    @staticmethod
    def clean_data():
        df = pd.read_csv(path("paper_to_abstract_with_doi.csv"))
        print(df.count())
        data_ = []
        for i, data in df.iterrows():
            if pd.isna(data["paper_doi"]) and pd.isna(data["paper_title"]):
                continue
            if not pd.isna(data["paper_title"]) and not pd.isna(data["title"]):
                if data["paper_title"] != data["title"]:
                    print(data["id"])
            data_.append((data["paper_id"], data["paper_doi"], data["paper_title"], data["abstract"]))

        new_df = pd.DataFrame(data_, columns=["paper_id", "paper_doi", "paper_title", "paper_abstract"])
        new_df.to_csv(path("paper_to_abstract_with_doi_v2.csv"), index=False)

        print(new_df.count())

    @staticmethod
    def _process_statements(df):
        all_statements = []
        for i, data in tqdm(df.iterrows(), total=len(df)):
            statement_id = data["id"]
            subject_id = ast.literal_eval(data["subject"])["id"]
            predicate_id = ast.literal_eval(data["predicate"])["id"]
            predicate_label = ast.literal_eval(data["predicate"])["label"]
            object_id = ast.literal_eval(data["object"])["id"]
            object_label = ast.literal_eval(data["object"])["label"]
            all_statements.append((statement_id, subject_id, predicate_id, predicate_label, object_id, object_label))

        return pd.DataFrame(all_statements,
                            columns=["statement_id", "subject_id", "predicate_id", "predicate_label", "object_id",
                                     "object_label"])

    @staticmethod
    def _is_in_abstract(df: pd.DataFrame):
        df["object_in_abstract"] = np.nan
        for i, data in df.iterrows():
            if pd.isna(data["paper_abstract"]) or pd.isna(data["object_label"]):
                continue

            elif str(data["object_label"]).lower() in str(data["paper_abstract"]).lower():
                df.at[i, "object_in_abstract"] = True
            else:
                df.at[i, "object_in_abstract"] = False

        return df


def main():
    p2ab = PapersToAbstracts()
    p2ab.merge_all_data()


if __name__ == '__main__':

    main()

