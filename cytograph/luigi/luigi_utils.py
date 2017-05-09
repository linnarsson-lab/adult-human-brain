from typing import *
import cytograph as cg


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
