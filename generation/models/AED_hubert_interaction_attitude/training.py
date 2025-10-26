import pytorch_lightning as pl
import utils.constants.constants as constants

class TrainModel():

    def __init__(self):
       super().__init__()
    
    def train_model(self, trainer_args, dm): 
        model = constants.model()
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model, dm)