import os

from orkg import ORKG


class OrkgClient:

    def __init__(self):
        self.connection = self.create_connection()

    @staticmethod
    def create_connection():
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

    def get_all_research_fields(self) -> list[tuple]:

        resources = []
        i = 0
        while True:
            # getting resources with class ResearchField
            response = self.connection.classes.get_resource_by_class("ResearchField",
                                                                     params={"sort": "id", "page": i, "size": 1000})

            for item in response.content:
                resources.append((item["id"], item["label"], item["classes"]))

            if len(response.content) == 0:
                resources.sort(key=lambda x: int(x[0][1:]))
                return resources

            # increasing the page number
            i = i + 1


# this section is just for testing
if __name__ == '__main__':
    client = OrkgClient()
    re = client.get_all_research_fields()
    for r in re:
        print(r)
