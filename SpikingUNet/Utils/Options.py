import wandb


class Param:
    def __init__(self):
        self.epoch = 30
        self.spiking_step = 25
        self.lr = 2e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.batch_size = 2

        self.classes = [i for i in range(13)]

        self.device = 'cuda:0'
        self.log = True
        self.project = 'DaconDomainAdaptive'
        self.log_name = 'SpikingUNet'

        self.root = 'D:/Side/Dacon/DomainAdaptive/dataset'
        self.run_type = ['train', 'val', 'test']