from typing import *
import cytograph as cg
import logging as lg
import luigi
import random
import string

lg.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=lg.DEBUG)


def EP2int(timepoint: str) -> int:
	if "P" in timepoint:
		return int(float(timepoint.lstrip("P"))) + 19
	else:
		return int(float(timepoint.lstrip("E")))


def time_check(tissue_name: str, time_par: str) -> bool:
	earlytime_s, latertime_s = time_par.split("-")
	try:
		tissue_earlytime_s, tissue_latertime_s = tissue_name.split("_")[-1].split("-")
	except ValueError:
		tissue_earlytime_s = tissue_name.split("_")[-1]
		tissue_latertime_s = tissue_earlytime_s
	earlytime, latertime = EP2int(earlytime_s), EP2int(latertime_s)
	tissue_earlytime, tissue_latertime = EP2int(tissue_earlytime_s), EP2int(tissue_latertime_s)
	return (earlytime <= tissue_earlytime) and (latertime >= tissue_latertime)


def logging(task: luigi.Task, log_dependencies: bool = False) -> lg.Logger:
	logger_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
	log_file = task.output().path + ".log"
	logger = lg.getLogger(logger_name)
	formatter = lg.Formatter('%(asctime)s %(levelname)s: %(message)s')
	fileHandler = lg.FileHandler(log_file, mode='w')
	fileHandler.setFormatter(formatter)
	# streamHandler = lg.StreamHandler()
	# streamHandler.setFormatter(formatter)

	logger.setLevel(lg.INFO)
	logger.addHandler(fileHandler)
	# logger.addHandler(streamHandler)

	if log_dependencies:
		logger.info("digraph G {")
		graph: Dict[str, List[str]] = {}

		def compute_task_graph(task: luigi.Task) -> None:
			name = task.__str__().split('(')[0]
			for dep in task.deps():
				if name in graph:
					graph[name].append(dep.__str__().split('(')[0])
				else:
					graph[name] = [dep.__str__().split('(')[0]]
				compute_task_graph(dep)

		compute_task_graph(task)
		for k, v in graph.items():
			for u in v:
				logger.info('"' + u + '" -> "' + k + '";')
		logger.info("}")
		logger.info("")

	for p in task.get_param_names():
		logger.info(f"{p} = {task.__dict__[p]}")
	logger.info("===")
	return logger


targets_map = {
	"All": [
		'Cephalic_E7-8', 'Forebrain_E9-11', 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
		'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18',
		'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18', 'Hindbrain_E9-11', 'Hindbrain_E12-15',
		'Hindbrain_E16-18'],
	"Forebrain": [
		"Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
		'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18'],
	"ForebrainDorsal": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18'],
	"ForebrainVentrolateral": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18'],
	"ForebrainVentrothalamic": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrothalamic_E16-18'],
	"Midbrain": ["Cephalic_E7-8", 'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18'],
	"Hindbrain": ["Cephalic_E7-8", 'Hindbrain_E9-11', 'Hindbrain_E12-15', 'Hindbrain_E16-18'],
	"Cortex": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18', "Cortex_P7"]}
