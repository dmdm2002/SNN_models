import numpy as np

from SpikingUNet.Utils.Metric import get_iou


# record: mIOU, Loss
class Displayer:
    def __init__(self, record):
        super().__init__()
        self.record = record
        self.scores = [0] * len(self.record)
        self.total_avg = [0] * len(self.record)
        self.count = 0

    def update(self, pred, gt, classes, loss):
        class_iou, class_weight = get_iou(pred, gt, classes)

        for i in range(len(class_iou)):
            self.scores[self.record[i]] += class_iou[i]

        self.scores[self.record.index('miou')] += class_iou.mean()
        self.scores[self.record.index('loss')] += loss.item()
        self.count += 1

    def average_score(self, epoch, disp=False, logging=False, db_runner=None, run_type='train'):
        for i, total_loss in enumerate(self.scores):
            self.total_avg[i] = total_loss / self.count
            if disp:
                print(f'{run_type} epoch[{epoch}] --> {self.record[i]}: {self.total_avg[i]}')
            if logging:
                self.logging(db_runner, self.record[i], self.total_avg[i], epoch, run_type)

    def logging(self, db_runner, name, value, epoch, run_type):
        db_runner.log({f'{run_type}_{name}': value}, step=epoch)

    def reset(self):
        self.count = 0
        self.scores = [0] * len(self.record)
        self.total_avg = [0] * len(self.record)
