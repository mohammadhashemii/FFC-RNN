from jiwer import wer

class HandwrittenRecognitionMetrics:
    def __init__(self):
        self.wer, self.ser = 0, 0
        self.n_smaples = 0


    def update_metric(self, sample: str, ground_truth: str):
        wer_sum = self.wer * self.n_smaples + wer(ground_truth, sample)
        ser_sum = self.ser * self.n_smaples + int(ground_truth != sample)

        self.n_smaples += 1
        self.wer = wer_sum / self.n_smaples
        self.ser = ser_sum / self.n_smaples




