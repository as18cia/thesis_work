import ast
import queue
import threading

import pandas as pd

from data_preparation.multi_thread_caller import MultiThreadCaller
from data_preparation.orkg_client import OrkgClient
from path_creator import path


class ReFieldsToPapers:

    def __init__(self):
        self.client = OrkgClient()
        self.caller = MultiThreadCaller()

    def get_research_field_to_papers(self):
        # todo: how about smart reviews and so on ???
        # getting all papers
        papers = self.client.get_resources_by_class("Paper")
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

        # turning the dictionary into a list of tuples
        research_field_to_papers_list = [(key, value["label"], len(value["papers"]), value["papers"]) for key, value in
                                         research_field_to_papers.items()]

        # fill in the rest of the Research fields
        all_research_fields = self.client.get_resources_by_class("ResearchField")
        for item in all_research_fields:
            if item[0] not in research_field_to_papers:
                research_field_to_papers_list.append((item[0], item[1], 0, []))

        # sorting by # of papers
        research_field_to_papers_list.sort(key=lambda x: x[2], reverse=True)

        # saving the results to a csv file
        df = pd.DataFrame(research_field_to_papers_list,
                          columns=["ResearchFieldId", "ResearchFieldLabel", "#OfPapers", "PaperIds"])
        df.to_csv(r"C:\Users\mhrou\Desktop\Orkg\ResearchFields_to_Papers.csv", index=False)

    @staticmethod
    def create_paper_to_re_mapping():
        df = pd.read_csv(path("ResearchFields_to_Papers.csv"))
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


if __name__ == '__main__':
    re = ReFieldsToPapers()
    s = re.create_paper_to_re_mapping()
