import pandas as pd
import stanza
from stanfordcorenlp import StanfordCoreNLP
import spacy as sp
from tqdm import tqdm

from path_creator import path


class Ner:

    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en')
        pass

    def get_pos_for_word(self, label):
        return self.nlp(label).sentences[0].words[0].pos

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
    n = Ner()
    print(n.get_pos_for_word("Global"))
    pass
