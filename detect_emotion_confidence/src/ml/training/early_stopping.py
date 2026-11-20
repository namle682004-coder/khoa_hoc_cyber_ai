class EarlyStopping:

    def __init__(self, patience=5):

        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.stop = False

    def step(self, val_score):

        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score <= self.best_score:

            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True

        else:

            self.best_score = val_score
            self.counter = 0

        return self.stop