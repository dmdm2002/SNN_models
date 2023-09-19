import torch
import torch.nn as nn
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
from SpikingUNet.Dataset.OneHot import label_to_one_hot_label

class Trainer(Param):
    def __init__(self):
        super().__init__()
        self.record_names = [i for i in self.classes]
        self.record_names.append('miou')
        self.record_names.append('loss')

        self.tr_disp = Displayer(record=self.record_names)
        self.te_disp = Displayer(record=self.record_names)

    def run(self):
        model = SpikingUNet(3, len(self.classes)).to(self.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr)
        # loss_fn = SF.ce_rate_loss()
        loss_fn = torch.nn.CrossEntropyLoss()

        if self.log:
            db_runner = wandb.init(project=self.project, name=self.log_name)
            p = Param()
            wandb.config.update(p.__dict__)
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
            for idx, (item, gt) in enumerate(tqdm(train_loader, desc=f'[Train Epoch: {ep}/{self.epoch}]')):
                item = item.float().to(self.device)
                gt = gt.long().to(self.device)

                model.train()
                logit = model(item)
                loss = loss_fn(logit, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.tr_disp.update(logit.detach().cpu(), gt.detach().cpu(), self.classes, loss)

            if self.log:
                self.tr_disp.average_score(ep, True, True, db_runner, 'train')
            else:
                self.tr_disp.average_score(ep, True, True, run_type='train')

            model.eval()
            with torch.no_grad():
                for idx, (item, gt) in enumerate(tqdm(val_loader, desc=f'[Test Epoch: {ep}/{self.epoch}]')):
                    item = item.to(self.device)
                    gt = gt.long().to(self.device)

                    logit = model(item)

                    loss = loss_fn(logit, gt)

                    self.te_disp.update(logit.detach().cpu(), gt.detach().cpu(), self.classes, loss)

            if self.log:
                self.te_disp.average_score(ep, True, True, db_runner, 'val')
            else:
                self.te_disp.average_score(ep, True, False, run_type='val')
