import ast
import pandas as pd

from data_preparation.multi_thread_caller import MultiThreadCaller
from data_preparation.orkg_client import OrkgClient


class ReFieldsToPapers:
    """"
    This class is to map the ORKG Research fields to the ORKG papers
    """

    def __init__(self):
        self.client = OrkgClient()
        self.caller = MultiThreadCaller()

    def get_research_field_to_papers(self):
        """"
        Creates the ResearchFields_to_Papers.csv file
        """
        # getting all papers from the orkg client
        papers = self.client.get_resources_by_class("Paper")
        # we only need the id here which is in the first position
        papers = [item[0] for item in papers]

        # getting statement for papers which contain predicate P30 -> ResearchField
        callable_fun = self.client.get_statement_based_on_predicate
        statements = self.caller.run(papers, callable_fun, 50, "P30")

        research_field_to_papers = {}
        for statement in statements:
            if statement["object"]["id"] in research_field_to_papers:
                research_field_to_papers[statement["object"]["id"]]["papers"].append(statement["subject"]["id"])
            else:
                research_field_to_papers[statement["object"]["id"]] = {
                    "label": statement["object"]["label"],
                    "papers": [statement["subject"]["id"]]
                }

        # turning the dictionary into a list of tuples (re_field id, re_field label, paper_ids)
        research_field_to_papers_list = [(key, value["label"], len(value["papers"]), value["papers"]) for key, value in
                                         research_field_to_papers.items()]

        # fill in the rest of the Research fields, the ones with no papers
        all_research_fields = self.client.get_resources_by_class("ResearchField")
        for item in all_research_fields:
            if item[0] not in research_field_to_papers:
                research_field_to_papers_list.append((item[0], item[1], 0, []))

        # sorting by # of papers
        research_field_to_papers_list.sort(key=lambda x: x[2], reverse=True)

        # saving the results to a csv file
        df = pd.DataFrame(research_field_to_papers_list,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "#OfPapers", "PaperIds"])
        df.to_csv("../data/processed/ResearchFields_to_Papers.csv", index=False)

    @staticmethod
    def create_paper_to_re_mapping():
        """"
        From the ResearchFields_to_Papers mapping creates a python dictionary for use by other functions
        """
        df = pd.read_csv("../data/processed/ResearchFields_to_Papers.csv")
        mapping = {}
        for i, data in df.iterrows():
            re_f_l = data["ResearchFieldLabel"]
            re_f_id = data["ResearchFieldId"]
            paper_ids = ast.literal_eval(data["PaperIds"])
            for p in paper_ids:
                if p not in mapping:
                    mapping[p] = {
                        "research_field_label": re_f_l,
                        "research_field_id": re_f_id
                    }
                else:
                    print("found one")
        return mapping

    def flatten_mapping(self):
        papers = self.client.get_resources_by_class("Paper")
        paper_id_to_title = {item[0]: item[1] for item in papers}
        df = pd.read_csv("../data/processed/ResearchFields_to_Papers.csv")

        mappings = []
        for i, data in df.iterrows():
            for paper_id in ast.literal_eval(data["PaperIds"]):
                mappings.append(
                    (data["ResearchFieldId"], data["ResearchFieldLabel"], paper_id, paper_id_to_title[paper_id]))

        df = pd.DataFrame(mappings,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "PaperId", "PaperTitle"])
        df.to_csv("../data/processed/ResearchFields_to_Papers_flattened.csv", index=False)


""""
The following section is for testing
"""
if __name__ == '__main__':
    re = ReFieldsToPapers()
    re.get_research_field_to_papers()
