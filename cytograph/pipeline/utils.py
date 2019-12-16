import os
import random


class Tempname:
	"""
	A context manager that generates a temporary pathname, which is
	renamed to its permanent name upon leaving the context. The renaming
	is guaranteed to be atomic at least on POSIX systems.
	"""
	def __init__(self, path: str) -> None:
		self.path = path
		self.temp_path = self.path + "_" + str(random.uniform(0, 1e6))
	
	def __enter__(self) -> str:
		return self.temp_path
	
	def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
		if os.path.exists(self.temp_path) and exc_type is None:
			os.rename(self.temp_path, self.path)
