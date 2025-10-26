import os
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.logger import MyLogger

import utils.constants.constants as constants

class Train():
    def __init__(self):
        super(Train, self).__init__()

        self.checkpoint_callback = self._create_checkpoint_callback()
        self.logger = MyLogger()
        self.trainer_args = self._create_trainer_args()
        self.dm = self._prepare_data_module()
        
        constants.train_model(self.trainer_args, self.dm)

    def _create_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=constants.saved_path,
            every_n_epochs=constants.log_interval,
            save_top_k=-1,
            save_weights_only=False,
            filename="{epoch}"
        )

    def _create_trainer_args(self):
        constants.number_of_gpu = int(os.environ["SLURM_GPUS_ON_NODE"]) * int(os.environ["SLURM_NNODES"])
        print("number of gpu", constants.number_of_gpu)

        trainer_args = {
            "accelerator": "gpu",
            "devices": int(os.environ["SLURM_GPUS_ON_NODE"]),
            "num_nodes": int(os.environ["SLURM_NNODES"]),
            "strategy": "ddp",
            "max_epochs": constants.n_epochs,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": constants.log_interval,
            "enable_progress_bar": False,
            "callbacks": [self.checkpoint_callback],
            "logger": self.logger,
            "amp_backend": "native",
            "sync_batchnorm": True,
            "enable_model_summary": True,
        }

        if constants.do_resume:
            checkpoint_path = self._find_last_checkpoint()
            if checkpoint_path:
                trainer_args["resume_from_checkpoint"] = checkpoint_path
                print("Resuming from checkpoint:", checkpoint_path)

        return trainer_args
    

    def _find_last_checkpoint(self):
        last_checkpoint = None
        max_epoch = -1

        for file in os.listdir(constants.saved_path):
            if file.endswith(".ckpt") and "epoch=" in file:
                try:
                    epoch = int(file.split("epoch=")[1].split(".")[0])
                    if epoch > max_epoch:
                        max_epoch = epoch
                        last_checkpoint = os.path.join(constants.saved_path, file)
                except (IndexError, ValueError):
                    continue

        return last_checkpoint


    def _prepare_data_module(self):
        dm = constants.customDataModule(stage="fit")
        dm.prepare_data()
        dm.setup(stage="fit")
        dm.is_prepared = True  # Avoid re-preparing
        return dm
        