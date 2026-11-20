from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, log_dir="runs/experiment"):

        self.writer = SummaryWriter(log_dir)

    def log_train(self, loss, epoch):

        self.writer.add_scalar("train/loss", loss, epoch)

    def log_val(self, loss, acc, epoch):

        self.writer.add_scalar("val/loss", loss, epoch)
        self.writer.add_scalar("val/accuracy", acc, epoch)

    def close(self):

        self.writer.close()