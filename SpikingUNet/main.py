from Runner.Train import Trainer
import wandb

if __name__ == '__main__':
    wandb.login()
    a = Trainer()
    a.run()