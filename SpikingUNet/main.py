from Runner.Train import Trainer
import wandb

if __name__ == '__main__':
    wandb.login(key='2f90610010c90fee7d521748a5c0d2d17db87255')
    a = Trainer()
    a.run()