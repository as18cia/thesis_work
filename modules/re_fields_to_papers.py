import queue
import threading

from tqdm import tqdm
import pandas as pd

from modules.orkg_client import OrkgClient


class ReFieldsToPapers:

    def __init__(self):
        self.client = OrkgClient()

    def get_research_field_to_papers(self):
        # todo: how about smart reviews and so on ???
        # getting all papers
        papers = self.client.get_resources_by_class("Paper")

        # getting statement for papers which contain predicate P30 -> ResearchField
        statements = self._get_statements_for_papers(papers)
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

    def _get_statements_for_papers(self, papers: list):
        in_q = queue.Queue()
        out_q = queue.Queue()
        for i, paper in enumerate(papers):
            in_q.put((i, paper))

        threads = [threading.Thread(target=self._work, args=(in_q, out_q)) for i in range(45)]

        # starting the threads
        for thread in threads:
            thread.start()

        # waiting for the threads to finish
        for thread in threads:
            thread.join()

        return list(out_q.queue)

    def _work(self, in_q: queue.Queue, out_q: queue.Queue):
        while not in_q.empty():
            item = in_q.get()
            index = item[0]
            out_q.put(self.client.get_statement_based_on_predicate(item[1][0], "P30"))
            print(index)


if __name__ == '__main__':
    re = ReFieldsToPapers()
    re.get_research_field_to_papers()
