"""Base worker class for the coordinator's per-client proxies.

Defines :class:`BaseWorker`, the shared infrastructure (serialisation,
optional zlib compression, retry-on-timeout) used by
:class:`WebsocketClientWorkerCustom` to communicate with one remote
client-worker process.
"""

import zlib

import torch,io
import binascii,time,ssl
import logging,websocket
TIMEOUT_INTERVAL = 60*60
# logger = logging.getLogger(__name__)
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


class BaseWorker:
	def __init__(self,usecompression:bool=False):
		self.usecompression=usecompression

	@staticmethod
	def create_worker_command_message(command_name: str, return_ids=None, *args, **kwargs):
		"""helper function creating a worker command message

		Args:
			command_name: name of the command that shall be called
			return_ids: optionally set the ids of the return values (for remote objects)
			*args:  will be passed to the call of command_name
			**kwargs:  will be passed to the call of command_name

		Returns:
			cmd_msg: a WorkerCommandMessage

		"""
		if return_ids is None:
			return_ids = []
		return WorkerCommandMessage(command_name, (args, kwargs, return_ids))

	def torchSeralize(self,tensor):
		buffer = io.BytesIO()
		torch.save(tensor, buffer)
		if self.usecompression:
			return zlib.compress(buffer.getvalue())
		else:
			return buffer.getvalue()

	def torchdeseralize(self,serialized_data):
		try:
			if self.usecompression:
				serialized_data=zlib.decompress(serialized_data)
			return torch.load(io.BytesIO(serialized_data))
		except:
			serialized_data = zlib.decompress(serialized_data)
			return torch.load(io.BytesIO(serialized_data))

	def seralize(self,data):
		return self.torchSeralize(data)

	def deseralize(self,data):
		return self.torchdeseralize(data)

	def _send_msg(self, message: bin, location=None) -> bin:
		return self._recv_msg(message)

	def _log_msgs_remote(self, value=True):
		return self._send_msg_and_deserialize("_log_msgs", value=value)

	@property
	def url(self):
		return f"wss://{self.host}:{self.port}" if self.secure else f"ws://{self.host}:{self.port}"

	def connect(self,maxtry=40,wakeupinterval=5):
		if self.ws is not None and self.ws.connected:
			self.logger.debug("Websocket Already connected %s", self.name)
			return
		args_ = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.url, "ping_interval": None}
		if self.secure:
			args_["sslopt"] = {"cert_reqs": ssl.CERT_NONE}
		self.wakeUp()
		for t in range(maxtry):
			try:
				self.logger.debug("Creating connection with %s ....", args_["url"])
				self.ws = websocket.create_connection(**args_)
				self.logger.debug("Connected to %s ....", args_["url"])
				break
			except Exception as e:
				self.logger.info("Exception in connection..... %s",t)
				if t%wakeupinterval==wakeupinterval-1:
					self.wakeUp()
					time.sleep(10)
				else:
					time.sleep(5)
		else:
			self.logger.critical("Not able to connecto to %s after %s tries",self.url,maxtry)
		self._log_msgs_remote(self.log_msgs)

	def close(self):
		self.ws.shutdown()

	def _forward_to_websocket_server_worker(self, message: bin) -> bin:
		self.ws.send(str(binascii.hexlify(message)))
		response = binascii.unhexlify(self.ws.recv()[2:-1])
		return response


	def _recv_msg(self, message: bin) -> bin:
		"""Forwards a message to the WebsocketServerWorker"""
		response = self._forward_to_websocket_server_worker(message)
		if not self.ws.connected:
			self.logger.warning("Websocket connection closed (worker: %s)", self.name)
			self.ws.shutdown()
			time.sleep(0.1)
			# Avoid timing out on the server-side
			self.ws = websocket.create_connection(self.url, max_size=None, timeout=TIMEOUT_INTERVAL)
			self.logger.warning("Created new websocket connection")
			time.sleep(0.1)
			response = self._forward_to_websocket_server_worker(message)
			if not self.ws.connected:
				raise RuntimeError(
					"Websocket connection closed and creation of new connection failed."
				)
		return response
