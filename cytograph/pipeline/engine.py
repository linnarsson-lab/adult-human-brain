import os
import shutil
import subprocess
import logging
from typing import *
from .punchcards import PunchcardDeck
from .workflow import process_subset, pool_leaves
from .config import config


class Engine:
	def __init__(self, deck: PunchcardDeck) -> None:
		self.deck = deck
		self.dryrun = config.execution.dryrun
	
	def build_execution_dag(self) -> Dict[str, List[str]]:
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
				stack.append(self.deck.get_subset(dep))
				tasks[s.longname()] = [dep]
			else:
				tasks[s.longname()] = []
		return tasks
	
	def execute(self) -> None:
		pass


# From https://stackoverflow.com/questions/52432988/python-dict-key-order-based-on-values-recursive-solution
def topological_sort(dependency_graph):
	# reverse the graph
	graph = {}
	for key, nodes in dependency_graph.items():
		for node in nodes:
			graph.setdefault(node, []).append(key)

	# init the indegree for each noe
	nodes = graph.keys() | set([node for adjacents in graph.values() for node in adjacents])
	in_degree = {node: 0 for node in nodes}

	# compute the indegree
	for k, adjacents in graph.items():
		for node in adjacents:
			in_degree[node] += 1

	# init the heap with the nodes with indegree 0 and priority given by key
	heap = [node for node, degree in in_degree.items() if degree == 0]

	top_order = []
	while heap:  # heap is not empty
		node = heap.pop()  # get the element with highest priority and remove from heap
		top_order.append(node)  # add to topological order
		for adjacent in graph.get(node, []):  # iter over the neighbors of the node
			in_degree[adjacent] -= 1
			if in_degree[adjacent] == 0:  # if the node has in_degree 0 add to the heap with priority given by key
				heap.append(adjacent)

	return top_order


class LocalEngine(Engine):
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck)

	def execute(self) -> None:
		tasks = self.build_execution_dag()
		logging.debug(tasks)

		# Figure out a linear execution order consistent with the DAG
		ordered_tasks = topological_sort(tasks)

		# Now we have the tasks ordered by the DAG, and run them
		if self.dryrun:
			logging.info("Dry run only, with the following execution plan")
		for ix, task in enumerate(ordered_tasks):
			if task == "_pool":
				if not self.dryrun:
					logging.info(f"\033[1;32;40mBuild step {ix + 1} of {len(ordered_tasks)}: cytograph pool\033[0m")
					subprocess.run(["cytograph", "--hide-message", "pool"])
				else:
					logging.info("cytograph pool")
			else:
				if not self.dryrun:
					logging.info(f"\033[1;32;40mBuild step {ix + 1} of {len(ordered_tasks)}: cytograph process {task}\033[0m")
					subprocess.run(["cytograph", "--hide-message", "process", task])
				else:
					logging.info(f"cytograph process {task}")


class CondorEngine(Engine):
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck)

	def execute(self) -> None:
		tasks = self.build_execution_dag()
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
