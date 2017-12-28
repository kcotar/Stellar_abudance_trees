from os import system
import multiprocessing


def python_run(command):
    print 'Running: ' + command
    system(command)

run_file = 'terminal_run_clusters_probable.txt'
# parse the file
txt = open(run_file, 'r')
txt_data = txt.readlines()
txt.close()
python_commands = [l[:-4] for l in txt_data if 'python NJ_tree_' in l]
print python_commands
print 'To be run:', len(python_commands)

# run selected commands
n_parallel = 5
pool = multiprocessing.Pool(n_parallel)
pool.map(python_run, python_commands)
pool.close()
