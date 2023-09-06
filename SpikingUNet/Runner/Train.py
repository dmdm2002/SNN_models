import torch
from torch.utils.data import DataLoader
import snntorch.functional as SF
import wandb

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from SpikingUNet.Model.SnnUNet import SpikingUNet
from SpikingUNet.Utils.Options import Param
from SpikingUNet.Utils.Display import Displayer
from SpikingUNet.Dataset.DaconSamsung import DaconSamsung


class Trainer(Param):
    def __init__(self):
        super().__init__()
        self.record_names = [i for i in self.classes]
        self.record_names.append('miou')
        self.record_names.append('loss')

        self.tr_disp = Displayer(record=self.record_names)
        self.te_disp = Displayer(record=self.record_names)

    def run(self):
        model = SpikingUNet(3, len(self.classes))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr)
        loss_fn = SF.ce_temporal_loss()

        if self.log:
            db_runner = wandb.init(project=self.project)
            wandb.config.update(Param.__dict__)
        else:
            db_runner = None

        transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ]
        )

        tr_dataset = DaconSamsung(self.root, self.run_type[0], transform, infer=False)
        val_dataset = DaconSamsung(self.root, self.run_type[1], transform, infer=False)
        train_loader = DataLoader(dataset=tr_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)

        for ep in range(self.epoch):
            for idx, (item, gt) in enumerate(tqdm(train_loader, desc=f'[Train Epoch: {self.epoch}/{ep}]')):
                item = item.to(self.device)
                gt = item.to(self.device)

                model.train()
                logit = model(item)
                loss = loss_fn(logit, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = logit.cpu().numpy()
                gt = gt.cpu().numpy()

                self.tr_disp.update(pred, gt, self.classes, loss)

            model.eval()
            with torch.no_grad():
                for idx, (item, gt) in enumerate(tqdm(val_loader, desc=f'[Test Epoch: {self.epoch}/{ep}]')):
                    item = item.to(self.device)
                    gt = gt.to(self.device)

                    logit = model(item)

                    loss = loss_fn(logit, gt)

                    pred = logit.cpu().numpy()
                    gt = gt.cpu().numpy()

                    self.te_disp.update(pred, gt, self.classes, loss)

            if self.log:
                self.tr_disp.average_score(ep, True, True, db_runner, 'train')
                self.te_disp.average_score(ep, True, True, db_runner, 'val')
            else:
                self.tr_disp.average_score(ep, True, True, run_type='train')
                self.te_disp.average_score(ep, True, False, run_type='val')
