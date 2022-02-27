from jiwer import wer
import jiwer


class HandwrittenRecognitionMetrics:
    def __init__(self):
        self.wer, self.ser = 0, 0
        self.n_smaples = 0
        self.n_words = 0

    def update_metric(self, sample: str, ground_truth: str):
        measures = compute_measures(ground_truth, sample)
        word_errors = self.wer * self.n_words + (measures.substitutions + measures.deletions + measures.insertions)
        ser_sum = self.ser * self.n_smaples + int(ground_truth != sample)

        self.n_smaples += 1
        self.n_words += len(ground_truth.split(" "))
        self.wer = word_errors / self.n_words
        self.ser = ser_sum / self.n_smaples

