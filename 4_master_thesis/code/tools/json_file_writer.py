##### save dictionary into json data ######
import json

def dict2json(dict_to_save, fname):
    f = open(fname, 'w')
    json.dump(dict_to_save, f, indent = 4, separators = (',', ':'))
    f.close()
