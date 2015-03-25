import random

from experiments.rep import Word2VecRepresentation
from nlpy.tasks.paraphrase.encoder import ParaphraseEncoder
from nlpy.tasks.paraphrase.classifier import FeaturePreProcessor


class SituationResponseSearcher(object):

    def __init__(self, corpus_list, rae_network, classify_network, vec=None, corpus_parser=None, parser=None):
        self._vec = vec if vec else Word2VecRepresentation()
        self.rae_network = rae_network
        self.classify_network = classify_network
        self._corpus_encoder = ParaphraseEncoder(rae_network, vec, corpus_parser)
        self._encoder = ParaphraseEncoder(rae_network, vec, parser)
        self._feature_processor = FeaturePreProcessor()
        self.load_data(corpus_list)

    def load_data(self, corpus_list):
        self.conversations = []
        self.sent_map = {}
        for corpus_path in corpus_list:
            content = open(corpus_path).read().strip()
            convs = content.split("\n\n")
            convs = [x.split("\n") + ["A: END"] for x in convs]
            conv_len = len(convs[0])
            for i in range(conv_len - 1):
                first_sent = convs[0][i]
                next_sent = convs[0][i + 1]
                if first_sent.startswith("B:") and next_sent.startswith("A:"):
                    for j in range(len(convs)):
                        sent = convs[j][i].split(":")[1].strip()
                        self.sent_map[sent] = (len(self.conversations), i)
            self.conversations.append(convs)

    def build_cache(self):
        self.rep_cache = {}
        for sent in self.sent_map:
            self.rep_cache[sent] = self._corpus_encoder.encode(sent)

    def search(self, sent, suggest_position=None):
        if not hasattr(self, "rep_cache"):
            self.build_cache()
        if suggest_position:
            suggest_conv, suggest_id = map(int, suggest_position.split(","))
        else:
            suggest_conv, suggest_id = -1, -1
        reps = self._encoder.encode(sent)
        max_prob = 0
        max_candidate = ""
        for candidate in self.sent_map:
            conv_id, sent_id = self.sent_map[candidate]
            bonus = 0.
            if conv_id == suggest_conv and sent_id == suggest_id:
                bonus = 0.5
            pooling_matrix = self._encoder.make_pooling_matrix(sent, candidate, reps, self.rep_cache[candidate])
            input_data = self._feature_processor.preprocess_nolabel(sent, candidate, pooling_matrix)
            prob = self.classify_network.predict(input_data)[0][1]
            prob += (1 - prob) * bonus
            if prob > max_prob:
                max_prob = prob
                max_candidate = candidate

        conv_id, sent_id = self.sent_map[max_candidate]
        convs = self.conversations[conv_id]
        response = random.choice([cv[sent_id+1] for cv in convs]).split(":")[1].strip()
        next = []
        if sent_id < len(convs[0]) - 2:
            next = [cv[sent_id+2].split(":")[1].strip() for cv in convs]
        print max_candidate, max_prob
        print response
        print "---"
        return {"response": response, "next": next, "position": "%d,%d" % (conv_id, sent_id+2)}

    @staticmethod
    def serve(param, searcher=None):
        if "input" not in param:
            return "no input"

        if not searcher:
            global situation_searcher
            if "situation_searcher" in globals():
                searcher = situation_searcher
            else:
                print "Loading searcher ..."
                situation_searcher = SituationResponseSearcher()
                searcher = situation_searcher

        input = param['input']

        output = searcher.search(input, suggest_position=param["position"])
        return output


