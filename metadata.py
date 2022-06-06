import requests
import time
import re
import urllib.parse

from fuzzywuzzy import fuzz

# TODO: This service should be part of the ORKG's back-end
TITLE_SIMILARITY_THRESHOLD = 90
SEMANTIC_SCHOLAR_REQUEST_RATE = 3


class MetadataService:
    __instance = None

    def __init__(self):
        if MetadataService.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            MetadataService.__instance = self

        self.crossRef = CrossRef()
        self.semanticScholar = SemanticScholar()

    @staticmethod
    def get_instance():
        if MetadataService.__instance:
            return MetadataService.__instance

        return MetadataService()

    def respect_rate_limits(func):

        def wrapper(self, *args, **kwargs):
            time.sleep(SEMANTIC_SCHOLAR_REQUEST_RATE)
            return func(self, *args, **kwargs)

        return wrapper

    @respect_rate_limits
    def by_doi(self, doi):
        return self.crossRef.by_doi(doi) or self.semanticScholar.by_doi(doi)

    @respect_rate_limits
    def by_title(self, title):
        return self.semanticScholar.by_title(title) or self.crossRef.by_title(title)


class CrossRef:

    @staticmethod
    def by_doi(doi):

        if not doi:
            return None

        url_encoded_doi = urllib.parse.quote_plus(doi)
        url = 'https://api.crossref.org/works/{}'.format(url_encoded_doi)

        response = requests.get(url)
        if not response.ok:
            return None

        response = response.json()

        if 'abstract' in response['message']:
            return Util.sanitize_abstract(response['message']['abstract'])

        return None

    @staticmethod
    def by_title(title):

        if not title:
            return None

        url_encoded_title = urllib.parse.quote_plus(title)
        url = 'https://api.crossref.org/works?rows=5&query.bibliographic={}'.format(url_encoded_title)

        response = requests.get(url)
        if not response.ok:
            return None

        response = response.json()

        doi = None
        if 'items' in response['message']:
            for item in response['message']['items']:
                if title.lower() == item['title'][0].lower():
                    doi = item['DOI']

            for item in response['message']['items']:
                if fuzz.ratio(title.lower(), item['title'][0].lower()) > TITLE_SIMILARITY_THRESHOLD:
                    doi = item['DOI']

        return CrossRef.by_doi(doi)


class SemanticScholar:

    @staticmethod
    def by_doi(doi):

        if not doi:
            return None

        url_encoded_doi = urllib.parse.quote_plus(doi)
        url = 'https://api.semanticscholar.org/v1/paper/{}'.format(url_encoded_doi)

        response = requests.get(url)
        if not response.ok:
            return None

        response = response.json()

        if 'abstract' in response:
            return Util.sanitize_abstract(response['abstract'])

        return None

    @staticmethod
    def by_title(title):

        if not title:
            return None

        url_encoded_title = urllib.parse.quote_plus(title)
        url = 'https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields=abstract,title'.format(
            url_encoded_title)

        response = requests.get(url)
        if not response.ok:
            return None

        response = response.json()

        if 'data' in response:
            for paper in response['data']:
                if title.lower() == paper['title'].lower():
                    return Util.sanitize_abstract(paper['abstract'])

            for paper in response['data']:
                if fuzz.ratio(title.lower(), paper['title'].lower()) > TITLE_SIMILARITY_THRESHOLD:
                    return Util.sanitize_abstract(paper['abstract'])

        return None


class Util:

    def sanitize_abstract(abstract):
        if not abstract:
            return None

        jats_regex = '</?jats:[a-zA-Z0-9_]*>'
        abstract = re.sub(jats_regex, ' ', abstract).strip()
        return ' '.join(abstract.split())
