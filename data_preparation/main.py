from data_preparation.finale_touches import FinalTouches
from data_preparation.papers_to_abstracts import PapersToAbstracts
from data_preparation.papers_to_contributions_to_statements import PaperToContributionToStatements
from data_preparation.re_fields_to_papers import ReFieldsToPapers
from data_preparation.split_data import SplitData


def main():
    # step 1: creating ** re_field <-> papers ** mapping (research field to list of corresponding papers)
    re = ReFieldsToPapers()
    re.get_research_field_to_papers()
    re.flatten_mapping()

    # step 2: creating ** re_field -> paper -> contribution -> statements
    to_contribution = PaperToContributionToStatements()
    to_contribution.get_contributions_for_papers()
    to_contribution.get_statements_for_contributions()

    # step 3: creating ** re_field <-> paper <-> abstract ** mapping
    to_abstract = PapersToAbstracts()
    to_abstract.get_abstracts_for_papers()

    # # step 4: categorizing object labels
    ft = FinalTouches()
    ft.finalize()

    # step 5: the last step is to split the data
    slt = SplitData()
    slt.split_training_and_evaluation()


if __name__ == '__main__':
    main()
