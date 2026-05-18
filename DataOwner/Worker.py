"""Base worker / command-message definitions shared by the data-owner side.

Defines :class:`WorkerCommandMessage`, the simple envelope object that the
:mod:`websocketServer` (de)serializes when the training coordinator sends
remote-procedure calls to a client worker.
"""

import logging
TIMEOUT_INTERVAL = 120
logger = logging.getLogger(__name__)
class WorkerCommandMessage:
    """Message used to execute a function of the remote worker."""
    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512
    def __init__(self, command_name: str, message: tuple):
        """Initialize a WorkerCommandMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the command properly.
        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.command_name = command_name
        self.message = message

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {(self.command_name, self.message)})"



