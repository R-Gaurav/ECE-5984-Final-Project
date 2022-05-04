import os

MAIN_DIR = "/home/rgaurav/projects/def-anarayan/rgaurav/ECE-5984-Final-Project/"
RESULTS_DIR = MAIN_DIR + "/exp_results/"

################################################################################
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR+"/circle_task/", exist_ok=True)
os.makedirs(RESULTS_DIR+"/frozen_lake_task/", exist_ok=True)
