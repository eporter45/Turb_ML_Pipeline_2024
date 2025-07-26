from Preprocessing.load_data_rst import load_data_rst_from_config

from Trials import TRIALS

trial = TRIALS['full_inter']
train_cases = trial['train']
test_cases = trial['test']
print('[DEBUG] train_cases :', train_cases)
print('[DEBUG] test_cases :', test_cases)


config = {"features": {'input': ['p', 'Ux', 'Uy'],
                       'output': 'rst'},
          "train_cases" : train_cases,
          "test_cases" : test_cases,


}

data = load_data_rst_from_config(config, train_cases, test_cases, path= '../Data')
