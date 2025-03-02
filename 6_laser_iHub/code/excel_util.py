# Excel utils

import os
import pandas as pd

class ExcelUtil():
    
    # ----------------------------
    # Convert the Excel files to csv 
    # -> specifically done for the 2023 iHub campaign
    # ----------------------------
    @staticmethod
    def excel2csv(path2read, path2save, *args, **kwargs):
        # --- Iterate over all files in the directiory
        for path in os.listdir(path2read):
            # check if current path is a proper file -> remove the files starting with "._'
            if path.startswith('._'):
                #print(f'exclude an unvalid file {path}')
                pass
            else:
                # (1) Read the excel file 
                fname, extension = path.split('.')
                df = pd.read_excel (f'{path2read}/{fname}.{extension}', sheet_name = 2, header = [0, 1], decimal=',')
                # (2) Modify the dataframe
                # Drop the laser emission collumn (= 3rd collumn)
                df.drop(columns=[df.keys()[3]], inplace=True)
                # Rename the key: merge the header rows into one
                new_keys = list(df.keys()[:-1].map('_'.join))
                # Modify the last key (otherwise the missing unit will create weird name)
                new_keys.append(f'{df.keys()[-1][0]}_V/s')
                # New key names
                df.columns = new_keys 
                # (3) Write the data to a csv
                df.to_csv(f'{path2save}/{fname}.csv', index=True, header=True) 