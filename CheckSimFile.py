import pandas as pd
import numpy as np

path_input = 'Output/'
file_input = 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet'

df = pd.read_parquet(path_input + file_input)