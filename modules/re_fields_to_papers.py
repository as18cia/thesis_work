import queue
import threading

import pandas as pd

from modules.multi_thread_caller import MultiThreadCaller
from modules.orkg_client import OrkgClient


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
        statements = self.caller.get_statements_for_papers(papers, callable_fun, 50, "P30")
        research_field_to_papers = {}
        for s in statements:
            for statement in s:
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


if __name__ == '__main__':
    re = ReFieldsToPapers()
    re.get_research_field_to_papers()
