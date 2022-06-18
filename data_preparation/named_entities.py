import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
import spacy as sp
from tqdm import tqdm

from path_creator import path


class Ner:

    def __init__(self):
        self.md = sp.load("en_core_web_trf")

    def map_label_to_entity_name(self):
        df = pd.read_csv(path("merged_statements_to_abstracts_v3.csv"))
        labels = df["object_label"].tolist()
        mapping = []
        StanfordCoreNLP

        # for l in tqdm(labels, total=len(labels)):
        #     ss = l.split(" ")
        #     if len(ss) > 1:
        #         mapping.append((l, "sentence"))
        #         continue
        #     elif len(ss) < 1:
        #         mapping.append((l, "None"))
        #         continue
        #     else:
        #         r = self.md(l)
        #         mapping.append((l, r[0].ent_type_))
        #
        # pd.DataFrame(mapping, columns=["l", "name"]).to_csv(path("testing.csv"), index=False)


if __name__ == '__main__':
    import stanza

    nlp = stanza.Pipeline('en')
    df = pd.read_csv(r"C:\Users\mhrou\Desktop\Orkg\merged_statements_to_abstracts_v3.csv")
    s = df["object_label"].tolist()
    s.sort(key=lambda x: len(x))
    re = []
    pos
    for x in tqdm(s, total=len(s)):
        new = x.strip()
        new = new.split(" ")
        if len(new) != 1:
            re.append((x, None))
            continue

        doc = nlp(new[0])
        if doc.entities:
            re.append((x, doc.entities[0].type))
        else:
            re.append((x, "not found"))

    pd.DataFrame(re, columns=["label", "ner"]).to_csv(path("labels.csv"), index=False)

    # import stanza
    # stanza.download('en')  # This downloads the English models for the neural pipeline
    # nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English
    # doc = nlp("Urkraine")
    # print(doc.entities[0].type)
    # pass
