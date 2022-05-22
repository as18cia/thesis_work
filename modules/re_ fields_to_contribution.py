import pandas as pd
import ast

from modules.multi_thread_caller import MultiThreadCaller
from modules.orkg_client import OrkgClient


class ReFieldsToContribution:

    def __init__(self):
        self.caller = MultiThreadCaller()
        self.orkg_client = OrkgClient()

    def research_field_to_contribution(self):
        # load the re_field to paper csv
        df = pd.read_csv(r"C:\Users\mhrou\Desktop\Orkg\ResearchFields_to_Papers.csv")

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
        statements = self.caller.get_statements_for_papers(papers, callable_fun, 50, "P31")
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

    def _create_re_field_to_contribution(self, df, paper_to_contribution):
        re_fi_to_con = []
        for i, data in df.iterrows():
            contributions = set()
            for paper in ast.literal_eval(data["PaperIds"]):
                contributions.update(paper_to_contribution[paper])
            re_fi_to_con.append(
                (data["ResearchFieldId"], data["ResearchFieldLabel"], len(contributions), list(contributions)))

        re_fi_to_con.sort(key=lambda x: x[2], reverse=True)
        return re_fi_to_con


if __name__ == '__main__':
    re = ReFieldsToContribution()
    re.research_field_to_contribution()
