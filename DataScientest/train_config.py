"""Serialisable training-configuration message.

Defines :class:`TrainConfig`, the configuration object the coordinator sends
to each client worker at the start of a round (model weights, optimiser
spec, scheduler spec, compression flag, ...). The model weights are
base64-encoded so the config can travel as plain JSON over WebSocket.
"""

import torch,io,base64


class TrainConfig:
    def __init__(
        self,
        model,
        device:str="cpu",
        batch_size: int = 32,
        optimizer: str = "SGD",
        optimizer_args: dict = {"lr": 0.1},
        scheduler_args:dict={"T_max":200,"last_epoch":-1},
        usecompression=False
    ):
        """Initializer for TrainConfig.

        Args:
            model: A traced torch nn.Module instance.
            loss_fn: A jit function representing a loss function which
                shall be used to calculate the loss.
            batch_size: Batch size used for training.
            epochs: Epochs used for training.
            optimizer: A string indicating which optimizer should be used.
            optimizer_args: A dict containing the arguments to initialize the optimizer. Defaults to {'lr': 0.1}.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the tensor.
            max_nr_batches: Maximum number of training steps that will be performed. For large datasets
                            this can be used to run for less than the number of epochs provided.
            shuffle: boolean, whether to access the dataset randomly (shuffle) or sequentially (no shuffle).
            loss_fn_id: The id_at_location of (the ObjectWrapper of) a loss function which
                        shall be used to calculate the loss. This is used internally for train config deserialization.
            model_id: id_at_location of a traced torch nn.Module instance (objectwrapper). . This is used internally for train config deserialization.
        """
        # syft related attributes

        # training related attributes
        self.model = model
        self.device=device
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler_args=scheduler_args
        self.usecompression=usecompression

    """TrainConfig abstraction.

    A wrapper object that contains all that is needed to run a training loop
    remotely on a federated learning setup.
    """

    def send(self,worker):
        buffer = io.BytesIO()
        if isinstance(self.model,torch.jit.ScriptModule):
            torch.jit.save(self.model, buffer)
        else:
            torch.save(self.model, buffer)
        encoded_model = base64.b64encode(buffer.getvalue()).decode('utf-8')
        worker.setTrainConfig(trainconfig={"model":encoded_model,"device":self.device,"batch_size":self.batch_size,"optimizer":self.optimizer,
                                           "optimizer_args":self.optimizer_args,"scheduler_args":self.scheduler_args,"usecompression":self.usecompression})