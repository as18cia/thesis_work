import re

import numpy as np
import pandas as pd
import spacy

class FinalTouches:

    def __init__(self):
        # make sure to download the pipeline first
        # python -m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _is_in_abstract(df: pd.DataFrame):
        df["ObjectInAbstract"] = np.nan
        for i, data in df.iterrows():
            if pd.isna(data["PaperAbstract"]) or pd.isna(data["ObjectLabel"]):
                continue

            elif str(data["ObjectLabel"]).lower() in str(data["PaperAbstract"]).lower():
                df.at[i, "ObjectInAbstract"] = True
            else:
                df.at[i, "ObjectInAbstract"] = False

        return df

    def finalize(self):
        df = pd.read_csv("../data/processed/ResearchField_to_papers_to_abstract_to_contribution_statements.csv")
        df = self._remove_unwanted_rows(df)
        df = self._removing_duplicates(df)
        df = self._is_in_abstract(df)
        df = self._drop_not_in_abstract(df)
        df = self._object_label_categorization(df)
        df.to_csv("../data/processed/finale_dataset.csv", index=False)

    @staticmethod
    def _remove_unwanted_rows(df):
        to_remove = []
        for i, data in df.iterrows():
            if len(str(data["ObjectLabel"]).strip()) < 3:
                to_remove.append(i)
                continue
            if str(data["ObjectLabel"]).isdigit():
                if int(str(data["ObjectLabel"])) < 999:
                    to_remove.append(i)
                    continue
        df = df.drop(df.index[to_remove])
        print("pruned " + str(len(to_remove)) + "rows")
        return df

    @staticmethod
    def _removing_duplicates(df):
        """
        we keep only one row with the same ('PaperId' and 'PredicateLabel' and 'ObjectLabel')
        """
        df = df.drop_duplicates(subset=['PaperId', 'PredicateLabel', 'ObjectLabel'], keep='last').reset_index(
            drop=True)
        return df

    def is_object_in_abstract(self, df):
        df["ObjectInAbstract"] = np.nan
        for i, data in df.iterrows():
            if pd.isna(data["PaperAbstract"]) or pd.isna(data["ObjectLabel"]):
                continue

            elif self._find_word_in_text(str(data["ObjectLabel"]))(data["PaperAbstract"]):
                df.at[i, "ObjectInAbstract"] = True
            else:
                df.at[i, "ObjectInAbstract"] = False
        return

    @staticmethod
    def _drop_not_in_abstract(df):
        to_drop = []
        for i, data in df.iterrows():
            if data["ObjectInAbstract"] is True:
                continue
            to_drop.append(i)

        df = df.drop(df.index[to_drop])
        return df

    @staticmethod
    def _find_word_in_text(w):
        return re.compile(r'\b({0})\b'.format(r'{}'.format(w)), flags=re.IGNORECASE).search

    @staticmethod
    def _object_label_categorization(df):
        df["Category"] = np.nan
        for i, data in df.iterrows():
            # cleaning
            label = data["ObjectLabel"]
            label = label.strip()

            # research problem
            if data["PredicateLabel"].strip().lower() == "has research problem":
                df.at[i, "Category"] = "research problem"
                break

            # years/date
            if data["ObjectLabel"].isdigit():
                if 1000 <= int(data["ObjectLabel"]) < 2100:
                    df.at[i, "Category"] = "year/date"
                    break

            # urls
            if label.startswith("http"):
                df.at[i, "Category"] = "url"
                break

            # location
            if data["PredicateLabel"].lower() in ["country, city, location, continent", "has location",
                                                  "study location", "countries"]:
                df.at[i, "Category"] = "location"
                break

            if label.lstrip('-').replace('.', '', 1).replace(',', '').isdigit():
                df.at[i, "Category"] = "number"
                break

            # todo: this might need some testing
            # check if unit measure
            s = label.split(" ")
            for a in s:
                if re.match("[0-9]+[.,]*[0-9]*", a):
                    df.at[i, "Category"] = "count/measurement"
                    break

            # checking for nouns / adjectives
            label_tokens = label.split(" ")
            if len(label_tokens) == 1:
                doc = nlp(label)
                pos = list(set([token.pos_ for token in doc]))[0]
                if pos == "ADJ":
                    df.at[i, "Category"] = "adj"
                    break
                if pos == "NOUN":
                    df.at[i, "Category"] = "noun"
                    break
                else:
                    df.at[i, "Category"] = "unkown"
                    break

            # checking for noun phrases and adj phrases


            # todo: this should be extended for phrases, acronyms and other categories

        return df

if __name__ == '__main__':
    import spacy

    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    # Process whole documents
    text = ("smart girl")
    doc = nlp(text)

    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "ADJ"])
    print("Verbs:", [token.pos_ for token in doc])

    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.doc)