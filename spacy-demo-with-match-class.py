from typing import NamedTuple


import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import DependencyMatcher
from spacy.util import compile_infix_regex
from spacy.symbols import ORTH, LEMMA, POS, TAG, DEP, IS_ALPHA

# Load spaCy model
nlp = spacy.load(
    "en_core_sci_lg",
    # enable=["lemmatizer", "tokenizer", "parser", "tagger", "attribute_ruler"],
)

infixes = [r"\."]
infix_regex = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer

new_word = "mg/dl"
nlp.vocab.strings.add(new_word)
special_case = [{ORTH: new_word}]
nlp.tokenizer.add_special_case(new_word, special_case)
nlp.vocab.strings.add(".")
special_case = [{ORTH: "."}]
nlp.tokenizer.add_special_case(".", special_case)
special_case = [{ORTH: "ng/mL"}, {ORTH: "."}]
nlp.tokenizer.add_special_case("ng/mL.", special_case)

# Define patterns for EntityRuler
patterns = [
    {"label": "LAB_RESULT", "pattern": [{"LOWER": {"in": ["positive", "negative"]}}]},
    {"label": "LAB_RESULT", "pattern": [{"LIKE_NUM": True}]},
    {"label": "LAB_UNIT", "pattern": [{"LOWER": {"in": ["mg/dl", "mmol/l"]}}]},
    {"label": "LAB_TYPE", "pattern": [{"LOWER": {"in": ["cholesterol", "glucose"]}}]},
]

# Create and add EntityRuler to pipeline
config = {
    "overwrite_ents": False,
}
ruler = nlp.add_pipe("entity_ruler", config=config, first=True)
ruler.add_patterns(patterns)

# Create DependencyMatcher and add patterns
matcher = DependencyMatcher(nlp.vocab)

# Unique reporting mechanism
reported_tokens = set()
# Updated dependency patterns to capture trios, duos, and single entities
dependency_patterns = [
    # Trio Pattern: LAB_RESULT connected to both LAB_UNIT and LAB_TYPE
    [
        {"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}},
        {
            "LEFT_ID": "lab_result",
            "REL_OP": ".",
            "RIGHT_ID": "lab_unit",
            "RIGHT_ATTRS": {
                "ENT_TYPE": "LAB_UNIT",
            },
        },
        {
            "LEFT_ID": "lab_result",
            "REL_OP": ";*",
            "RIGHT_ID": "lab_type",
            "RIGHT_ATTRS": {
                "ENT_TYPE": "LAB_TYPE",
            },
        },
    ],
    # Duo Patterns: LAB_RESULT and LAB_UNIT
    [
        {"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}},
        {
            "LEFT_ID": "lab_result",
            "REL_OP": ".",
            "RIGHT_ID": "lab_unit",
            "RIGHT_ATTRS": {
                "ENT_TYPE": "LAB_UNIT",
            },
        },
    ],
    # LAB_RESULT and LAB_TYPE
    [
        {"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}},
        {
            "LEFT_ID": "lab_result",
            "REL_OP": ".*",
            "RIGHT_ID": "lab_type",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"},
        },
    ],
    # LAB_UNIT and LAB_TYPE
    [
        {"RIGHT_ID": "lab_unit", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_UNIT"}},
        {
            "LEFT_ID": "lab_unit",
            "REL_OP": ".*",
            "RIGHT_ID": "lab_type",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"},
        },
    ],
    # Single Patterns: LAB_RESULT, LAB_UNIT, LAB_TYPE
    [{"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}}],
    [
        {
            "RIGHT_ID": "lab_unit",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_UNIT"},
        }
    ],
    [{"RIGHT_ID": "lab_type", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"}}],
]

matcher.add("lab_dependency", dependency_patterns)

# Process the text
text = "The cholesterol level is 200 mg/dL. The glucose result is 120 mmol/L. The test came back positive"
doc = nlp(text)
print([x for x in doc])
# Extract and report entities uniquely
lab_entities = {"LAB_RESULT": set(), "LAB_UNIT": set(), "LAB_TYPE": set()}

matches = []
# Matching dependency patterns
for match_id, token_ids in matcher(doc):
    match = []
    for key in token_ids:
        entity = doc[key]

        if entity.i not in reported_tokens:
            lab_entities[entity.ent_type_].add(entity.text)
            reported_tokens.add(entity.i)
            match.append((entity.ent_type_, entity.text))
        else:
            match = None
            break
    if match:
        matches.append(match)
        print(match)
        print("\n\n")

print(matches)


# Define a NamedTuple for match details
class Match(NamedTuple):
    label: str
    annotated_text: str
    candidate_text: str
    start_index: int
    end_index: int

    @property
    def unique_id(self):
        # Unique identifier based on start and end index
        return (self.start_index, self.end_index)


# # Define a class for match handling
# class Match:
#     def __init__(self, match_details: MatchDetails):
#         self.details = match_details
#         self.fields = ['label', 'annotated_text', 'candidate_text', 'start_index', 'end_index']

#     def __getitem__(self, item):
#         return getattr(self.details, item, getattr(self, item))


# Define a class for grouping matches
class MatchGroup:
    def __init__(self, matches: list[Match], doc):
        self.doc = doc
        self.unique_match_ids = set()
        self.matches = matches
        self.unique_id_ = set([x.unique_id for x in matches])

    def to_token_labels(self):
        # Method to convert matches to a list of token labels
        token_labels = len(self.doc) * ["O"]
        for match in self.matches:
            start, end = match.start_index, match.end_index
            for i in range(start, end):
                token_labels[i] = match.label
        return token_labels

    def to_token_text_label_tuples(self):
        # Method to convert matches to a list of token labels
        tuples = [(token, "O") for token in self.doc]
        for match in self.matches:
            start, end = match.start_index, match.end_index
            for i in range(start, end):
                tuples[i] = (match.annotated_text, match.label)
        return tuples

    def __iter__(self):
        return self.matches.__iter__()

    @property
    def unique_id(self):
        # Unique identifier based on start and end index
        return frozenset(self.unique_id_)


class MatchGroupList(list):
    def to_combined_label_list(self):
        if not self:
            return []

        # Initialize the combined label list with 'O's
        combined_labels = len(self[0].doc) * ["O"]

        for match_group in self:
            token_labels = match_group.to_token_labels()
            for i, label in enumerate(token_labels):
                if label != "O":
                    combined_labels[i] = label

        return combined_labels

    def to_combined_token_label_tuple_list(self):
        if not self:
            return []

        # Initialize the combined label list with 'O's
        combined_labels = [(token, "O") for token in doc]

        for match_group in self:
            token_label_tuples = match_group.to_token_text_label_tuples()
            for i, tuple in enumerate(token_label_tuples):
                if tuple[1] != "O":
                    combined_labels[i] = tuple

        return combined_labels


reported_tokens = set()
reported_matches = set()

# Modify the logic for extracting and reporting entities
lab_entities = {"LAB_RESULT": set(), "LAB_UNIT": set(), "LAB_TYPE": set()}
matches = []
match_groups = []

for match_id, token_ids in matcher(doc):
    matches = []
    for key in token_ids:
        entity = doc[key]
        reported_tokens.add(entity.text)
        lab_entities[entity.ent_type_].add(entity.text)
        # Create a MatchDetails instance and add it to matches
        match_details = Match(
            label=entity.ent_type_,
            annotated_text=entity.text,
            candidate_text=doc.text,
            start_index=entity.i,
            end_index=entity.i + 1,
        )
        if match_details.unique_id in reported_tokens:
            matches = []
            break
        reported_tokens.add(match_details.unique_id)
        matches.append(match_details)

    if matches:
        match_group = MatchGroup(matches, doc)
        if match_group.unique_id not in reported_matches:
            reported_matches.add(match_group.unique_id)
            match_groups.append(match_group)


# Example usage of MatchGroup
print("Token labels from matches:")
[
    [print("\t" + str(match)) for match in match_group.to_token_text_label_tuples()]
    for match_group in match_groups
]

match_group_list = MatchGroupList()
match_group_list.extend(match_groups)
print(match_group_list.to_combined_label_list())
print(match_group_list.to_combined_token_label_tuple_list())

spacy.displacy.serve(doc, style="dep", auto_select_port=True)
spacy.displacy.serve(doc, style="ent", auto_select_port=True)
