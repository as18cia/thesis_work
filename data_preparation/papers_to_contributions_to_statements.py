import datetime
import pandas as pd

from data_preparation.multi_thread_caller import MultiThreadCaller
from data_preparation.orkg_client import OrkgClient


class PaperToContributionToStatements:

    def __init__(self):
        self.client = OrkgClient()
        self.caller = MultiThreadCaller()

    def get_contributions_for_papers(self):
        """
        for each paper in the csv file from last step we fetch the contributions
        """
        df = pd.read_csv("../data/processed/ResearchFields_to_Papers_flattened.csv")
        papers = list(set(df["PaperId"].tolist()))
        callable_fun = self.client.get_statement_based_on_predicate
        contributions = self.caller.run(papers, callable_fun, 50, "P31")
        print(datetime.datetime.now())

        papers_to_contributions = {}
        for item in contributions:
            if item["subject"]["id"] not in papers_to_contributions:
                papers_to_contributions[item["subject"]["id"]] = set()
                papers_to_contributions[item["subject"]["id"]].add(item["object"]["id"])
            else:
                papers_to_contributions[item["subject"]["id"]].add(item["object"]["id"])

        mappings = []
        for i, data in df.iterrows():
            if data["PaperId"] not in papers_to_contributions:
                print(data["PaperId"] + " does not have a Contribution")
                continue

            contributions = papers_to_contributions[data["PaperId"]]
            for c in contributions:
                d = [i for i in data]
                d.append(c)
                mappings.append(d)

        df = pd.DataFrame(mappings,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "PaperId",
                                   "PaperTitle", "Contribution"])
        df.to_csv("../data/processed/ResearchField_to_papers_to_Contribution.csv", index=False)

    def get_statements_for_contributions(self):
        """
        from each contribution we fetch the related statements
        """
        df = pd.read_csv("../data/processed/ResearchField_to_papers_to_Contribution.csv")
        contributions = list(set(df["Contribution"].tolist()))
        callable_fun = self.client.get_statement_based_on_predicate
        statements = self.caller.run(contributions, callable_fun, 50)
        contributions_to_statements = {}

        for statement in statements:
            if statement["subject"]["id"] not in contributions_to_statements:
                contributions_to_statements[statement["subject"]["id"]] = []
                contributions_to_statements[statement["subject"]["id"]].append(
                    (statement["predicate"]["label"], statement["object"]["label"]))
            else:
                contributions_to_statements[statement["subject"]["id"]].append(
                    (statement["predicate"]["label"], statement["object"]["label"]))
        mappings = []
        for i, data in df.iterrows():
            if data["Contribution"] not in contributions_to_statements:
                print(data["Contribution"] + " does not have a statement")
                continue

            statements = contributions_to_statements[data["Contribution"]]
            for s in statements:
                d = [i for i in data]
                d.append(s[0])
                d.append(s[1])
                mappings.append(d)

        df = pd.DataFrame(mappings,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "PaperId",
                                   "PaperTitle", "Contribution", "PredicateLabel", "ObjectLabel"])
        df.to_csv("../data/processed/ResearchField_to_papers_to_contribution_statements.csv", index=False)
