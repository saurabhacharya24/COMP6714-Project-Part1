# Import Libraries and Modules here...
import spacy
from math import log
from itertools import chain, combinations


class InvertedIndex:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        # You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}

    # Your implementation for indexing the documents...
    def index_documents(self, documents):

        nlp = self.nlp

        token_dict = {}
        entity_dict = {}

        for doc_id, text in documents.items():
            doc_text = nlp(text)

            for token in doc_text:
                if (not token.is_stop) and (not token.is_punct):
                    if token.text not in token_dict:
                        token_dict[token.text] = {doc_id: 1}
                    else:
                        curr_doc_obj = token_dict[token.text]
                        if doc_id not in curr_doc_obj:
                            curr_doc_obj[doc_id] = 1
                        else:
                            curr_doc_obj[doc_id] += 1

            for ent in doc_text.ents:
                if ent.text not in entity_dict:
                    entity_dict[ent.text] = {doc_id: 1}
                else:
                    curr_doc_obj = entity_dict[ent.text]
                    if doc_id not in curr_doc_obj:
                        curr_doc_obj[doc_id] = 1
                    else:
                        curr_doc_obj[doc_id] += 1

            final_tokens = {key: value for key, value in token_dict.items()}

        for key, value in token_dict.items():
            if key in entity_dict.keys():
                ent = entity_dict[key]

                checksum = []
                for docid, count in ent.items():
                    if docid in value.keys():
                        if (value[docid] > 1) and (docid not in checksum):
                            value[docid] -= count
                            checksum.append(docid)
                        else:
                            value.pop(docid, None)

                if not value:
                    final_tokens.pop(key)

        idf_toks = {}

        for k, v in final_tokens.items():

            for doc, count in v.items():
                if count != 0:
                    v[doc] = 1.0 + log(1.0 + log(count))
                else:
                    v[doc] = 0

            idf = 1.0 + log(len(documents) / (1.0 + len(v)))
            idf_toks[k] = idf

        self.tf_tokens = final_tokens
        self.idf_tokens = idf_toks

        idf_ents = {}

        for k, v in entity_dict.items():

            for doc, count in v.items():
                v[doc] = 1.0 + log(count)

            idf = 1.0 + log(len(documents) / (1.0 + len(v)))
            idf_ents[k] = idf

        self.tf_entities = entity_dict
        self.idf_entities = idf_ents

    # Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        nlp = self.nlp

        query_tokens = nlp(Q)
        token_list = []

        for token in query_tokens:
            token_list.append(str(token))

        query_splits = []

        query_splits.append((token_list, []))

        doe_list = [key for key, value in DoE.items()]

        subsets = list(powerset(token_list))
        final_subsets = []

        for x in subsets:
            val = ' '.join(map(str, x))
            final_subsets.append(val)

        final_ents = []

        for sub in final_subsets:
            final_tokens = []
            removal_tokens = []

            if sub in doe_list:
                final_ents.append(sub)

                for token in token_list:
                    if str(token) in sub and str(token) not in removal_tokens:
                        removal_tokens.append(str(token))

                    final_tokens.append(str(token))

                while len(removal_tokens) != 0:
                    popped = removal_tokens.pop(0)
                    final_tokens.remove(popped)

                query_splits.append((final_tokens, [sub]))

        final_splits = []

        for split in query_splits:
            if split not in final_splits:
                final_splits.append(split)

        joined_splits = final_splits

        for i in range(0, len(final_splits)):
            curr_tokens = final_splits[i][0]
            curr_ent = final_splits[i][1]

            for other_ent in range((i + 1), len(final_splits)):
                if other_ent != i:
                    next_ent = final_splits[other_ent][1]
                    if len(curr_ent) != 0:
                        for j in range(0, len(curr_ent)):
                            curr_entity = curr_ent[j]
                            next_entity = next_ent[j]

                            checksum = ""

                            for l in range(0, len(next_entity.split(" "))):
                                if next_entity.split(" ")[l] in curr_tokens:
                                    checksum += next_entity.split(" ")[l] + " "

                            updated_tokens = curr_tokens.copy()
                            checksum = checksum.strip()
                            if checksum == next_entity:
                                for word_num in range(0, len(checksum.split(" "))):
                                    updated_tokens.remove(checksum.split(" ")[word_num])

                                joined_splits.append((updated_tokens, [curr_entity, next_entity]))

        splits = []

        token_check = []
        for s in joined_splits:
            if s not in splits and s[0] not in token_check:
                token_check.append(s[0])
                splits.append(s)

        return splits

    # Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):

        tf_tokens = self.tf_tokens
        idf_tokens = self.idf_tokens

        tf_entities = self.tf_entities
        idf_entities = self.idf_entities

        scores = []

        for split in query_splits:
            tokens_score_total = 0.0
            entities_score_total = 0.0

            for token in split[0]:
                idf_score = 0.0
                tf_score = 0.0

                if str(token) in tf_tokens:
                    token_in_tf = tf_tokens[str(token)]

                    for doc, tf in token_in_tf.items():
                        if doc == doc_id:
                            tf_score = tf

                if str(token) in idf_tokens:
                    idf_score = idf_tokens[str(token)]

                tokens_score_total += tf_score * idf_score

            for entity in split[1]:
                idf_score = 0
                tf_score = 0

                if entity in tf_entities:
                    ent_in_tf = tf_entities[entity]

                    for doc, tf in ent_in_tf.items():
                        if doc == doc_id:
                            tf_score = tf

                if entity in idf_entities:
                    idf_score = idf_entities[entity]

                entities_score_total += tf_score * idf_score

            final_score = entities_score_total + (0.4 * tokens_score_total)

            token_ent_object = {
                'tokens': split[0],
                'entities': split[1]
            }

            scores.append((final_score, token_ent_object))

        max_val = 0.0
        max_score_tuple = ()

        for score_tuple in scores:
            if float(score_tuple[0]) > max_val:
                max_val = float(score_tuple[0])
                max_score_tuple = score_tuple

        return max_score_tuple


# Credit-Python Docs: https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
