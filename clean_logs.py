import pandas as pd
import os

df = pd.read_csv('log_files.csv', names=[
                 'run_id', 'model', 'dataset', 'prompt', 'shots', 'promptset'])

df2 = pd.read_csv('log_files_backup.csv', names=[
    'run_id', 'model', 'dataset', 'prompt', 'shots', 'promptset'])

# iterate over filenames in logs folder
for filename in os.listdir('logs'):
    # if filename is not in the run_id column of the dataframe
    if filename[:-6] not in df['run_id'].values and filename[:-6] not in df2['run_id'].values:
        # delete the file
        os.remove(f'logs/{filename}')
