import os

from orkg import ORKG


class OrkgClient:

    def __init__(self):
        self.connection = self._create_connection()

    @staticmethod
    def _create_connection():
        return ORKG(host="https://www.orkg.org/orkg/", creds=(os.environ["orkg_email"], os.environ["orkg_password"]))

    def get_all_resources(self):
        resources = []
        i = 0
        while True:
            o = self.connection.resources.get(params={"sort": "id", "page": i, "size": 1000})
            for item in o.content:
                resources.append((item["id"], item["label"], item["classes"]))

            # if there are no more resources we return the list of resources
            if len(o.content) == 0:
                return resources

            # increasing the page number
            i = i + 1

    def get_resources_by_class(self, resource_class: str) -> list[tuple]:
        # resources can be Paper or ResearchField for example
        resources = []
        i = 0
        while True:
            # getting resources with class ResearchField
            response = self.connection.classes.get_resource_by_class(resource_class,
                                                                     params={"sort": "id", "page": i, "size": 200})

            for item in response.content:
                if item["id"] == "Custom_ID":
                    continue
                resources.append((item["id"], item["label"], item["classes"]))

            if len(response.content) == 0:
                resources.sort(key=lambda x: int(x[0][1:]))
                return resources

            # increasing the page number
            i = i + 1

    def get_statement_based_on_predicate(self, subject: str, predicate_id: str = None):
        statements = []

        i = 0
        while True:
            # getting resources with class ResearchField
            response = self.connection.statements.get_by_subject(subject,
                                                                 params={"sort": "id", "page": i, "size": 1000})
            for item in response.content:
                statements.append(item)

            if len(response.content) == 0:
                break
            i = i + 1

        filtered_statements = []
        if predicate_id:
            for statement in statements:
                if statement["predicate"]["id"] == predicate_id:
                    filtered_statements.append(statement)
            return filtered_statements

        return statements

    def get_doi_for_paper(self, paper):
        res = self.get_statement_based_on_predicate(paper, "P26")
        doi = set()
        for r in res:
            doi.add(r["object"]["label"])
        if len(doi) == 0:
            return None
        elif len(doi) == 1:
            return doi.pop()
        else:
            print("more than one doi")
            return doi.pop()

    def get_all_statements(self):
        statements = []
        i = 0
        while True:
            # getting resources with class ResearchField
            response = self.connection.statements.get(params={"sort": "id", "page": i, "size": 1000})

            for item in response.content:
                statements.append((item["id"], item["subject"], item["predicate"], item["object"]))

            if len(response.content) == 0:
                statements.sort(key=lambda x: int(x[0][1:]))
                return statements

            # increasing the page number
            i = i + 1


# this section is just for testing
if __name__ == '__main__':
    client = OrkgClient()
    re = client.get_doi_for_paper("R3046")
