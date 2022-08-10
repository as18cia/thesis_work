import pandas as pd
import ast

from data_preparation.multi_thread_caller import MultiThreadCaller
from data_preparation.orkg_client import OrkgClient


class ReFieldsToContribution:
    """"
    This class is to map the ORKG Research fields to the ORKG Contributions, provided a mapping from Research fields
    to ORKG papers already exists
    Please check ./re_fields_to_papers.py for the script to create a mapping from Research fields to ORKG papers
    """

    def __init__(self):
        self.caller = MultiThreadCaller()
        self.orkg_client = OrkgClient()

    def research_field_to_contribution(self):
        """"
        Creates the mapping from the ORKG Research fields to the ORKG Contributions
        """
        # load the re_field to paper csv
        df = pd.read_csv("../data/ResearchFields_to_Papers.csv")

        # get relevant papers
        papers = set()
        for i, data in df.iterrows():
            papers.update(ast.literal_eval(data["PaperIds"]))
        papers = list(papers)

        # get paper to contribution mapping
        paper_to_contribution = self._get_paper_to_contribution(papers)

        # create re_field to contribution
        re_field_to_contribution_df = self._create_re_field_to_contribution(df, paper_to_contribution)

        # save the csv file
        df = pd.DataFrame(re_field_to_contribution_df,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "#OfContributions", "Contributions"])
        df.to_csv(r"C:\Users\mhrou\Desktop\Orkg\ResearchFields_to_Contributions.csv", index=False)

    def _get_paper_to_contribution(self, papers):
        paper_to_contribution = {}
        callable_fun = self.orkg_client.get_statement_based_on_predicate
        statements = self.caller.run(papers, callable_fun, 50, "P31")
        for statement in statements:
            for s in statement:
                if s["subject"]["id"] in paper_to_contribution:
                    paper_to_contribution[s["subject"]["id"]].append(s["object"]["id"])
                else:
                    paper_to_contribution[s["subject"]["id"]] = [s["object"]["id"]]

        # ensuring there are no duplicates
        for k, v in paper_to_contribution.items():
            paper_to_contribution[k] = list(set(v))

        # adding papers with no contribution
        for p in papers:
            if p not in paper_to_contribution:
                paper_to_contribution[p] = []

        return paper_to_contribution

    @staticmethod
    def _create_re_field_to_contribution(df, paper_to_contribution):
        re_fi_to_con = []
        for i, data in df.iterrows():
            contributions = set()
            for paper in ast.literal_eval(data["PaperIds"]):
                contributions.update(paper_to_contribution[paper])
            re_fi_to_con.append(
                (data["ResearchFieldId"], data["ResearchFieldLabel"], len(contributions), list(contributions)))

        re_fi_to_con.sort(key=lambda x: x[2], reverse=True)
        return re_fi_to_con


""""
The following section is for testing
"""
if __name__ == '__main__':
    re = ReFieldsToContribution()
    re.research_field_to_contribution()
