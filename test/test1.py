import pandas as pd
import zipfile

zf = zipfile.ZipFile('stage_2_private_solution.csv.7z') # having First.csv zipped file.
df = pd.read_csv(zf.open('a.csv'))