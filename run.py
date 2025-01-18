import os
import subprocess
import sys
def run(command):
    subprocess.call(command, shell=True)
def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, time_cost = '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'args:' in line:
            parameter = line
        if 'final best performance:' in line:
            result = line
        if 'Experiment cost:' in line:
            time_cost = line
    return parameter, result, time_cost

def process_sample():
    for mmd_weight in [5,4,3,2,1,0.1,0]:
        for laplacian_target_weight in [0,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4]:
            for k in [1,2,3,4,5,6,7,8,9,10,20]:
              for ratio in [0,0.3,0.5,0.7,0.9,1]:
                path=os.path.join('output/xxx', "x-%s-%s-%s-%s"%(mmd_weight ,laplacian_target_weight,k,ratio))
                cmd = 'python main.py ' +' '+ \
                     ' --output_dir ' + str(path) + ' '+ \
                     ' --mmd_weight ' + str(mmd_weight) + ' '+ \
                     ' --laplacian_target_weight  ' + str(laplacian_target_weight ) + ' '+ \
                    ' --ratio ' + str(ratio) + ' ' + \
                     ' --k ' + str(k)
                run(cmd)
                sys.stdout.flush()
if __name__ == '__main__':
    process_sample()






