import os
import argparse
from tools.utils import parse_config



parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='generation',
                    choices=['generation', 'analysis', 'visualization'], help='task')
parser.add_argument('--exp', type=str, default='base', help='experiment',
                    choices=['base', 'number', 'scalefree_dist_number', 'randomIdx', 'coupling_strength', 'random_num',
                             'scalefree_dist', 'mouse_chimera', 'mouse_connect', 'mouse_control'])

if __name__ == '__main__':
    args = parser.parse_args()
    task = args.task
    exercise = args.exp

    dataFolder = os.path.join(os.getcwd(), 'data')
    outputFolder = os.path.join(dataFolder, 'output')
    analysisFolder = os.path.join(dataFolder, 'analysis')
    figFolder = os.path.join(dataFolder, 'figures')

    config = parse_config(os.path.join(os.getcwd(), 'config', 'paras_' + exercise + '.yaml'))
    config['outputFolder'] = outputFolder
    config['analysisFolder'] = analysisFolder
    config['figFolder'] = figFolder
    config['exp'] = exercise

    if task == 'generation':
        import tasks.generation as gen
        # make data folder
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        print('Generation room: {}'.format(exercise))
        genTask = eval('gen.' + exercise + '_generation')(config)
        genTask.run()

    elif task == 'analysis':
        import tasks.analysis as ana
        # check the sequence of the task
        if not os.path.exists(os.path.join(outputFolder, exercise)):
            raise ValueError('Please generate the data first!')
        if not os.path.exists(analysisFolder):
            os.makedirs(analysisFolder)
        print('Analysis exp: {}'.format(exercise))
        processTask = eval('ana.' + exercise + "_analysis")(config)
        processTask.run()

    elif task == 'visualization':
        import tasks.visualization as vis

        # check the sequence of the task
        gen_do = os.path.exists(os.path.join(outputFolder, exercise))
        ana_do = os.path.exists(analysisFolder)
        if not gen_do:
            raise ValueError('Please generate the data first!')
        if gen_do and (not ana_do):
            raise ValueError('You have generated the data, but not analysed yet!')
        if (not gen_do) and ana_do:
            raise ValueError('There are no generated data but analysed data. Check the data whether is paired!')
        if not os.path.exists(figFolder):
            os.makedirs(figFolder)
        print('Visualization exp: {}'.format(exercise))
        plotTask = eval('vis.' + exercise + '_visualization')(config)
        plotTask.run()

    else:
        raise ValueError('Task must be generation, evaluation or visualization!')