from typing import NamedTuple, Any, List
import spacy
from spacy.matcher import DependencyMatcher
from spacy.util import compile_infix_regex
from spacy.symbols import ORTH, LEMMA, POS, TAG, DEP, IS_ALPHA
from abc import ABC, abstractmethod
from collections import defaultdict


class Model(ABC):
    @abstractmethod
    def predict(self) -> Any:
        """
        Abstract method to be overridden in subclasses.
        Should return a prediction in a flexible format (list, tensor, etc.).
        """
        pass


class NERModel(Model):
    def predict(self) -> List[str]:
        """
        Override of the abstract predict method.
        Specifically returns a list of strings as predictions.
        """
        pass


# Define a NamedTuple for match details
class Match(NamedTuple):
    label: str
    annotated_text: str
    candidate_text: str
    start_index: int
    end_index: int
    match_id: int

    @property
    def unique_id(self):
        # Unique identifier based on start and end index
        return (self.start_index, self.end_index)


class MatchGroup:
    def __init__(self, matches: list[Match], doc):
        self.doc = doc
        self.unique_match_ids = set()
        self.matches = matches
        self.unique_id_ = set([x.unique_id for x in matches])

    def to_token_labels(self):
        token_labels = len(self.doc) * ["O"]
        for match in self.matches:
            start, end = match.start_index, match.end_index
            for i in range(start, end):
                token_labels[i] = match.label
        return token_labels

    def to_token_text_label_tuples(self):
        tuples = [(token, "O") for token in self.doc]
        for match in self.matches:
            start, end = match.start_index, match.end_index
            for i in range(start, end):
                tuples[i] = (match.annotated_text, match.label)
        return tuples

    def __str__(self):
        return "\n".join([str(x) for x in self.matches])

    def __iter__(self):
        return self.matches.__iter__()

    @property
    def unique_id(self):
        return frozenset(self.unique_id_)


class MatchGroupList(list):
    def to_combined_label_list(self):
        if not self:
            return []

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

        combined_labels = [(token, "O") for token in self[0].doc]

        for match_group in self:
            token_label_tuples = match_group.to_token_text_label_tuples()
            for i, tuple in enumerate(token_label_tuples):
                if tuple[1] != "O":
                    combined_labels[i] = tuple

        return combined_labels

    def print_vertical_groups(self):
        print("Token labels from matches:")
        [
            (
                [print("\t" + str(match)) for match in match_group.to()],
                print("\n"),
            )
            for match_group in self
        ]

    def print_vertical_groups(self):
        print("Token labels from matches:")
        [
            (
                [
                    print("\t" + str(match))
                    for match in match_group.to_token_text_label_tuples()
                ],
                print("\n"),
            )
            for match_group in self
        ]

    def __str__(self):
        return "\n\n".join([str(x) for x in self])


class RuleBasedTokenClassifier(NERModel):
    def __init__(self, model="en_core_sci_lg", disable: List = ["ner"]):
        self.nlp = spacy.load(model, disable=disable)
        config = {
            "overwrite_ents": False,
        }
        self.ruler = self.nlp.add_pipe("entity_ruler", config=config)
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self.span_ruler = self.nlp.add_pipe(
            "span_ruler",
            first=True,
            config={"annotate_ents": True, "overwrite": True},
        )
        self.ent_types = set()
        self.dep_types = set()

    def add_entity_ruler_patterns(self, patterns):
        self.ruler.add_patterns(patterns)
        for pattern in patterns:
            self.ent_types.add(pattern["label"])

    def add_dependency_matcher_patterns(self, pattern_name, patterns):
        self.matcher.add(pattern_name, patterns)
        for pattern in patterns:
            for element in pattern:
                self.dep_types.add(element["RIGHT_ID"])

    def add_span_patterns(self, patterns):
        """
        Adds patterns to the SpanMatcher component.
        :param patterns: A list of patterns to be added to the SpanMatcher.
        """
        self.span_ruler.add_patterns(patterns)
        for pattern in patterns:
            self.ent_types.add(pattern["label"])

    def process_text(self, text):
        doc = self.nlp(text)

        matches = self.matcher(doc)

        lab_entities = defaultdict(set)

        matches = []
        for match_id, token_ids in self.matcher(doc):
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

        return lab_entities, doc

    def process_text_with_match_groups(self, text):
        doc = self.nlp(text)

        reported_tokens = set()
        reported_matches = set()

        matches = []
        match_groups = []

        for match_id, token_ids in self.matcher(doc):
            matches = []
            for key in token_ids:
                entity = doc[key]
                reported_tokens.add(entity.text)
                match_details = Match(
                    label=entity.ent_type_,
                    annotated_text=entity.text,
                    candidate_text=doc.text,
                    start_index=entity.i,
                    end_index=entity.i + 1,
                    match_id=match_id,
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
        match_group_list = MatchGroupList()
        match_group_list.extend(match_groups)
        return match_group_list

    def predict(self, text):
        doc = self.nlp(text)

        entity_ruler_matches = []
        for entity in doc.ents:
            match_details = Match(
                label=entity.label_,
                annotated_text=entity.text,
                candidate_text=doc.text,
                start_index=entity.start,
                end_index=entity.end,
                match_id=-1,
            )
            entity_ruler_matches.append(match_details)

        match_group_list = MatchGroupList()
        match_group_list.append(MatchGroup(entity_ruler_matches, doc))
        return match_group_list.to_combined_label_list()

    def add_tokenizer_infix_pattern(self, pattern):
        infixes = pattern
        infix_regex = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer

    def add_tokenizer_special_case(self, word, case):
        self.nlp.vocab.strings.add(word)
        self.nlp.tokenizer.add_special_case(word, case)

    def add_word(self, word):
        self.nlp.vocab.strings.add(word)
        self.matcher = DependencyMatcher(self.nlp.vocab)

    def add_special_case(self, word, case):
        self.nlp.tokenizer.add_special_case(word, case)

    def display_deps(self, text):
        spacy.displacy.serve(self.nlp(text), style="dep", auto_select_port=True)


processor = RuleBasedTokenClassifier()
processor.add_word("mg/dl")
processor.add_tokenizer_infix_pattern(r"\.")
processor.add_tokenizer_special_case("mg/dL.", [{ORTH: "mg/dL"}, {ORTH: "."}])
processor.add_tokenizer_special_case("mmol/L.", [{ORTH: "mmol/L"}, {ORTH: "."}])

patterns = [
    {"label": "LAB_RESULT", "pattern": [{"LOWER": {"in": ["positive", "negative"]}}]},
    {"label": "LAB_RESULT", "pattern": [{"LIKE_NUM": True}]},
    {"label": "LAB_UNIT", "pattern": [{"LOWER": {"in": ["mg/dl", "mmol/l"]}}]},
    {"label": "LAB_TYPE", "pattern": [{"LOWER": {"in": ["cholesterol", "glucose"]}}]},
]

processor.add_entity_ruler_patterns(patterns)

reported_tokens = set()
dependency_patterns = [
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
    [
        {"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}},
        {
            "LEFT_ID": "lab_result",
            "REL_OP": ".*",
            "RIGHT_ID": "lab_type",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"},
        },
    ],
    [
        {"RIGHT_ID": "lab_unit", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_UNIT"}},
        {
            "LEFT_ID": "lab_unit",
            "REL_OP": ".*",
            "RIGHT_ID": "lab_type",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"},
        },
    ],
    [
        {"RIGHT_ID": "B-diagnosis", "RIGHT_ATTRS": {"ENT_TYPE": "diagnosis"}},
        {
            "LEFT_ID": "B-diagnosis",
            "REL_OP": ".",
            "RIGHT_ID": "I-diagnosis",
            "RIGHT_ATTRS": {"ENT_TYPE": "diagnosis"},
        },
    ],
    [{"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_RESULT"}}],
    [
        {
            "RIGHT_ID": "lab_unit",
            "RIGHT_ATTRS": {"ENT_TYPE": "LAB_UNIT"},
        }
    ],
    [{"RIGHT_ID": "lab_type", "RIGHT_ATTRS": {"ENT_TYPE": "LAB_TYPE"}}],
    [{"RIGHT_ID": "lab_result", "RIGHT_ATTRS": {"ENT_TYPE": "diagnosis"}}],
]

processor.add_dependency_matcher_patterns("lab_dependency", dependency_patterns)
patterns = [
    {"label": "diagnosis", "pattern": [{"lower": "malignant"}, {"lower": "melanoma"}]},
]
processor.add_span_patterns(patterns)

text = "The cholesterol level is 200 mg/dL. The glucose result is 120 mmol/L. The test came back positive. The patient has malignant melanoma."

print("Match Groups")
print(processor.process_text_with_match_groups(text))
print("\n\n")
prediction = processor.predict(text)
print("Single list of entity types")
print(prediction)
