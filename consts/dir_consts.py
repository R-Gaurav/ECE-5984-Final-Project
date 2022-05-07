import os

MAIN_DIR = "/scratch/rgaurav/ece_5984_drl_final_project/"
RESULTS_DIR = MAIN_DIR + "/exp_results/"

################################################################################
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR+"/circle_task/", exist_ok=True)
os.makedirs(RESULTS_DIR+"/frozen_lake_task/", exist_ok=True)
