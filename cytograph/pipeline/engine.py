import sys
import os
import shutil
import subprocess
import logging
import yaml
from typing import *
from .punchcards import PunchcardDeck
from .config import config, merge_config, ExecutionConfig


class Engine:
	'''
	An execution engine, which takes a :class:`PunchcardDeck` and calculates an execution plan in the form
	of a dependency graph. The Engine itself does not actually execute the graph. This is the job of
	subclasses such as :class:`LocalEngine`` and :class:`CondorEngine`, which take the execution plan and
	executes it in some manner (e.g. locally and serially, or on a cluster and in parallel).
	'''

	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		self.deck = deck
		self.dryrun = dryrun
	
	def build_execution_dag(self) -> Dict[str, List[str]]:
		"""
		Build an execution plan in the form of a dependency graph, encoded as a dictionary.

		Returns:
			Dictionary mapping tasks to their dependencies
	
		Remarks:
			The tasks are named for the punchcard subset they involve (using long subset names),
			and the pooling task is denoted by the special task name '_pool'.
		"""
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
				dep_subset = self.deck.get_subset(dep)
				if dep_subset is None:
					logging.error(f"Dependency '{dep}' of '{s.longname()}' was not found in punchcard deck.")
					sys.exit(1)
				stack.append(dep_subset)
				tasks[s.longname()] = [dep]
			else:
				tasks[s.longname()] = []
		return tasks
	
	def execute(self) -> None:
		pass


# From https://stackoverflow.com/questions/52432988/python-dict-key-order-based-on-values-recursive-solution
def topological_sort(dependency_graph: Dict[str, List[str]]) -> List[str]:
	"""
	Sort the dependency graph topologically, i.e. such that dependencies are
	listed before the tasks that depend on them.
	"""
	# reverse the graph
	graph: Dict[str, List[str]] = {}
	for key, deps in dependency_graph.items():
		for node in deps:
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
	"""
	An execution engine that executes tasks serially and locally.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

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
	"""
	An engine that executes tasks in parallel on a HTCondor cluster, using the DAGman functionality
	of condor. Tasks will be executed in parallel as much as possible while respecting the
	dependency graph.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

	def execute(self) -> None:
		tasks = self.build_execution_dag()
		# Make job files
		exdir = os.path.join(config.paths.build, "condor")
		if not os.path.exists(exdir):
			os.mkdir(exdir)
		for task in tasks.keys():
			cmd = ""
			excfg: Optional[ExecutionConfig] = None
			# Get the right execution configuration for the task (CPUs etc.)
			if task == "_pool":
				cfg_file = os.path.join(config.paths.build, "pool_config.yaml")
				if os.path.exists(cfg_file):
					merge_config(config, cfg_file)
				excfg = config.execution
				cmd = "pool"
			else:
				subset = self.deck.get_subset(task)
				if subset is None:
					logging.error(f"Subset '{task}' not found among punchcards.")
					sys.exit(1)
				config.execution.merge(subset.execution)
				excfg = config.execution
				cmd = f"process {task}"

			# Generate the condor submit file for the task
			with open(os.path.join(exdir, task + ".condor"), "w") as f:
				f.write(f"""
getenv       = true
executable   = {os.path.abspath(shutil.which('cytograph'))}
arguments    = "{cmd}"
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
				f.write(f"JOB {task} {os.path.join(exdir, task)}.condor DIR {config.paths.build}\n")
			for task, deps in tasks.items():
				if len(deps) == 0:
					continue
				f.write(f"PARENT {' '.join(deps)} CHILD {task}\n")

		if not self.dryrun:
			logging.info(f"condor_submit_dag {os.path.join(exdir, '_dag.condor')}")
			subprocess.run(["condor_submit_dag", os.path.join(exdir, "_dag.condor")])
		else:
			logging.info(f"(Dry run) condor_submit_dag {os.path.join(exdir, '_dag.condor')}")

# TODO: SlurmEngine using job dependencies (https://hpc.nih.gov/docs/job_dependencies.html)
# TODO: SgeEngine using job dependencies (https://arc.leeds.ac.uk/using-the-systems/why-have-a-scheduler/advanced-sge-job-dependencies/)
