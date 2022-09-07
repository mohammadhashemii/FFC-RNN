from jiwer import wer
import jiwer


class HandwrittenRecognitionMetrics:
    def __init__(self):
        self.wer, self.ser, self.cer = 0, 0, 0
        self.n_smaples = 0
        self.n_words = 0
        self.n_chars = 0

    def update_metric(self, sample: str, ground_truth: str, char_based=False):
        if not char_based:
            measures = jiwer.compute_measures(ground_truth, sample)
            word_errors = self.wer * self.n_words + (
                        measures['substitutions'] + measures['deletions'] + measures['insertions'])
            ser_sum = self.ser * self.n_smaples + int(ground_truth != sample)

            self.n_smaples += 1
            self.n_words += len(ground_truth.split(" "))
            self.wer = word_errors / self.n_words
            self.ser = ser_sum / self.n_smaples
        else:
            measures = jiwer.compute_measures(ground_truth, sample)
            word_errors = self.wer * self.n_words + (
                    measures['substitutions'] + measures['deletions'] + measures['insertions'])

            cer_sum = self.cer * self.n_chars + jiwer.cer(ground_truth, sample) * len(list(ground_truth))

            self.n_chars += len(list(ground_truth))
            self.n_words += len(ground_truth.split(" "))
            self.wer = word_errors / self.n_words
            self.cer = cer_sum / self.n_chars