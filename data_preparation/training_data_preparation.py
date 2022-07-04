import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preparation.named_entities import Ner
from data_preparation.orkg_client import OrkgClient
from data_preparation.re_fields_to_papers import ReFieldsToPapers
from path_creator import path
import time
import ast


class PapersToAbstracts:
    """
    This is the main module in this folder.
    This module contains multiple functions that do the following:
        * mapping orkg papers to their doi
        * retrieving statements with contributions as subject id
        * retrieving abstracts for papers
        * merging the data
        * cleaning the data
        * ...
    """

    def __init__(self):
        # self.client = OrkgClient()
        self.classificator = Ner()
        # self.papers_to_re = ReFieldsToPapers().create_paper_to_re_mapping()

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
    def get_abstracts_for_papers():
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

    @staticmethod
    def removing_duplicates():
        df = pd.read_csv(path("merged_statements_to_abstracts_v3_pruned.csv"))
        df = df.drop_duplicates(subset=['paper_id', 'predicate_label', 'object_label'], keep='last').reset_index(
            drop=True)
        df = df.drop_duplicates(subset=['paper_id', 'predicate_label', 'object_id'], keep='last').reset_index(drop=True)
        df.to_csv(path("merged_statements_to_abstracts_v3_pruned_deduplicated.csv"), index=False)

    def ner(self):
        df = pd.read_csv(path("worldcities.csv"))
        cities = [x.lower() for x in df["city"].tolist()]
        df = pd.read_csv(path("contries.csv"))
        contries = [x.lower() for x in df["Name"].tolist()]
        locations = cities + contries

        df = pd.read_csv(path("first_iter.csv"))
        to_remove = []
        for i, data in tqdm(df.iterrows(), total=len(df)):
            if len(data["object_label"].strip()) < 3:
                to_remove.append(i)
                continue
            # if True:  # pd.isna(data["ner"]):
            #     ner = self._ner(data["object_label"], data["predicate_label"], locations)
            #     if ner:
            #         df.at[i, "ner"] = ner
            #         continue
            #
            #     pos = self.pos_for_label(data["object_label"])
            #     if pos:
            #         df.at[i, "ner"] = pos
            #         continue
            #
            #     phrase_type = self.phrase_type(data["object_label"])
            #     if phrase_type:
            #         df.at[i, "ner"] = phrase_type
            #         continue
        df = df.drop(df.index[to_remove])
        df.to_csv(path("first_iter.csv"), index=False)

    def _ner(self, label: str, predicate: str, locations: list):
        # todo: adj phrases to noun phrases

        # cleaning
        label = label.strip()

        # research problem
        if predicate.strip().lower() == "has research problem":
            return "research problem"

        # number
        if label.isnumeric():
            if len(label) == 4:
                # check if year
                if label[0] in ["1", "2"]:
                    return "year"
        if self.is_number(label):
            return "number"

        # acronyms
        if re.fullmatch(r'[A-Z]+', string=label):
            return "acronym"

        # urls
        if label.startswith("http"):
            return "url"

        # locations
        if label.lower() in locations or predicate.lower() in ["country, city, location, continent", "has location",
                                                               "study location", "countries"]:
            return "location"

        # check if unit measure
        s = label.split(" ")
        for i in s:
            # if self.is_number(i):
            #     return "possible unit"
            if re.match("[0-9]+[.,]*[0-9]*", i):
                return "count/measurement"

    @staticmethod
    def is_number(label):
        if label.startswith("-"):
            label = label[1:]
        try:
            float(label)
            return True
        except:
            pass

        try:
            int(label)
            return True
        except:
            pass
        return False

    def pos_for_label(self, label: str):
        labels = label.strip().lower().split(" ")
        if len(labels) != 1:
            return
        ner = self.classificator.get_pos_for_word(labels[0])
        if ner == "PROPN":
            return "NOUN"
        return ner

    def phrase_type(self, phrase):
        phrase = phrase.strip().lower().split(" ")
        cleaner = []
        for l in phrase:
            if len(l) > 2:
                cleaner.append(l)

        if len(cleaner) > 4:
            return

        poses = []
        for l in cleaner:
            poses.append(self.classificator.get_pos_for_word(l))

        if poses:
            if poses[-1] in ["PROPN", "NOUN"]:
                return "noun phrase"

            if poses[-1] == "ADJ":
                return "adjective phrase"

    def cleaning_pass(self):
        df = pd.read_csv(path("data_set.csv"))

        re_fields = df["research_field_label"].tolist()
        counts = []
        for x in set(re_fields):
            counts.append((x, re_fields.count(x)))

        counts.sort(key=lambda x: x[1], reverse=True)
        resutl = [x[0] for x in counts if x[1] < 2]
        print(resutl)
        # to_delte = []
        # for i, data in df.iterrows():
        #     if data["research_field_label"] in resutl:
        #         to_delte.append(i)
        #
        # df = df.drop(df.index[to_delte])
        # df.to_csv(path("data_set.csv"), index=False)


def main():
    p2ab = PapersToAbstracts()
    p2ab.cleaning_pass()


if __name__ == '__main__':
    # main()

    df = pd.read_csv(path("data_set.csv"))
    ner = df["ner"].tolist()
    counts = []
    for x in set(ner):
        counts.append((x, ner.count(x)))

    counts.sort(key=lambda x: x[1], reverse=True)

    xx = []
    for x in counts:
        xx.append((x[0], x[1], str(round(x[1] / 5930 * 100, 1)) + "%"))

    print(pd.DataFrame(xx, columns=["Category", "# of occurrences", "percentage in dataset"]))
