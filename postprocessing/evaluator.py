import time
import operator
import numpy as np
import sys

from sklearn.metrics import classification_report, precision_recall_fscore_support


class Evaluator(object):
    def __init__(self, name, label2id, pure_labels=None):
        self.name = name
        self.label2id = label2id

        self.cost_sum = 0.0
        self.avg_fscore = 0.0
        self.true = []
        self.pred = []
        self.start_time = time.time()
        self.total_samples = 0
        self.report = None

        if pure_labels is None:
            self.pure_labels = self.label2id
        else:
            self.pure_labels = pure_labels

    def append_data(self, cost, pred_label_ids, true_label_ids):
        self.total_samples += 1
        self.cost_sum += cost
        self.pred.extend(pred_label_ids)
        self.true.extend(true_label_ids)

    def remove_nones(self):
        none_idx = [i for i, l in enumerate(self.true) if l is None]
        for idx in sorted(none_idx, reverse=True):
            del self.pred[idx]
            del self.true[idx]

        assert len(self.pred) == len(self.true)

        assert None not in self.pred
        assert None not in self.true

    def gen_results(self):

        assert len(self.true) == len(self.pred) != 0
        target = sorted(self.pure_labels, key=lambda k: self.pure_labels[k])
        print(target)
        self.remove_nones()

        print(len(self.pred))
        self.report = classification_report(self.true, self.pred, labels=range(len(target)), target_names=target, digits=6)
        _, _, self.avg_fscore, _ = precision_recall_fscore_support(self.true, self.pred, average="weighted")
        return self.report

    def print_results(self):
        print("{0}_total_samples: {1}".format(self.name, self.total_samples))
        print("{0}_cost_sum: {1}".format(self.name, self.cost_sum))
        print("{0}_Classification Report".format(self.name))
        print(self.report)

    def write_results(self, filename, text, spec='a'):
        with open(filename, spec, encoding='utf-8') as f:
            f.write("-" * 40 + "\n")
            f.write(text + "\n")
            f.write("-" * 40 + "\n")
            f.write("{0}_total_samples: {1}\n".format(self.name, self.total_samples))
            f.write("{0}_cost_sum: {1}\n".format(self.name, self.cost_sum))
            f.write("{0}_Classification Report\n".format(self.name))
            f.write(self.report)

    def verify_results(self):
        if np.isnan(self.cost_sum) or np.isinf(self.cost_sum):
            sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
            exit()
