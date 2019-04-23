import os
import shutil
import subprocess
import logging
from .punchcards import PunchcardDeck
from .workflow import process_subset, pool_leaves
from .config import config


class Engine:
	def __init__(self, deck: PunchcardDeck) -> None:
		self.deck = deck
		self.dryrun = config.execution.dryrun
	
	def build_execution_dag(self) -> None:
		stack = self.deck.root.get_leaves()
		if len(stack) > 1:
			tasks = {"_pool": [s.longname() for s in stack]}
		else:
			tasks = {}
		while len(stack) > 0:
			s = stack.pop()
			if s.longname() in tasks:
				continue
			dep = s.dependency()
			if dep is not None:
				stack.append(dep)
				tasks[s] = [dep]
			else:
				tasks[s] = []
		return tasks
	
	def execute(self) -> None:
		pass


class LocalEngine(Engine):
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck)

	def execute(self) -> None:
		tasks = self.build_execution_dag(self.deck)
		# Figure out a linear execution order consistent with the DAG
		ordered_tasks = []

		def add_task(t, deps):
			if deps == []:
				ordered_tasks.append(t)
			else:
				for d in deps:
					add_task(d, tasks[d])

		start, deps = tasks.items().next()
		add_task(start, deps)

		# Now we have the tasks ordered by the DAG, and run them
		for task in ordered_tasks:
			if task == "_pool":
				if not self.dryrun:
					pool_leaves(deck)
				else:
					logging.info("cytograph pool")
			else:
				if not self.dryrun:
					process_subset(deck.get_subset(task))
				else:
					logging.info(f"cytograph process {task}")


class CondorEngine(Engine):
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck)

	def execute(self) -> None:
		tasks = self.build_execution_dag(self.deck)
		# Make job files
		exdir = os.path.join(config.paths.build, "execution")
		if not os.path.exists(exdir):
			os.mkdir(exdir)
		for task in tasks.keys():
			if task != "_pool":
				excfg = self.deck.get_subset(task).execution_config
			else:
				excfg = self.deck.pool_execution_config
			with open(os.path.join(exdir, task + ".condor"), "w") as f:
				f.write(f"""
getenv       = true
executable   = {os.path.abspath(shutil.which('cytograph'))}
arguments    = "process {task}"
log          = {os.path.join(exdir, task)}.log
output       = {os.path.join(exdir, task)}.out
error        = {os.path.join(exdir, task)}.error
request_cpus = {excfg.n_cpus}
request_gpus = {excfg.n_gpus}
request_memory = {excfg.memory * 1024}
queue 1\n
""")

		with open(os.path.join(exdir, "_dag.condor"), "w") as f:
			for task in tasks.keys():
				f.write(f"JOB {task} {task}.condor DIR {config.paths.build}\n")
			for task, deps in tasks.items():
				f.write(f"PARENT {' '.join(deps)} CHILD {task}\n")

		if not self.dryrun:
			subprocess.run(["condor_submit_dag", os.path.join(exdir, "_dag.condor")])
		else:
			logging.info(f"condor_subbmit_dag {os.path.join(exdir, '_dag.condor')}")

# TODO: SlurmEngine using job dependencies (https://hpc.nih.gov/docs/job_dependencies.html)
# TODO: SgeEngine using job dependencies (https://arc.leeds.ac.uk/using-the-systems/why-have-a-scheduler/advanced-sge-job-dependencies/)
