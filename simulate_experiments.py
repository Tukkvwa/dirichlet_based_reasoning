import numpy as np
import matplotlib.pyplot as plt
from CocktailSortRTModel import CocktailSortRunTimeModel
from MergeSortRTModel import MergeSortRunTimeModel
from Sorter import merge_sort, cocktail_sort
from SortingProblemGenerator import SortingProblemGenerator
from MetaCognitiveSortingAgent import MetaCognitiveSortingAgent
from SCADSSortingAgent import SCADSSortingAgent
# Placeholders for agent classes and generator

def create_experiments(nr_subjects : int, nr_trials: int):
    """
    Create learning trials for nr_subjects number of subjects.
    Returns:
        experiments: dict of experiments for each subject
        test_trials: list of test trials
    """
    experiments = {}
    for s in range(nr_subjects):
        experiments[s] = {}
        experiments[s]["duration"] = 0 #duration of learning tials, this is a placeholder
        experiments[s]["learning_trials"] = {}
        for t in range(nr_trials):
            experience = {
                'algorithm': algorithm_params[t],
                'features': feature_params[t],
                'problem': generator.generate_problem([feature_params[t][0], feature_params[t][1]/feature_params[t][0]],  0.1/1000),
            }
            if algorithm_params[t]==1:
                experience['run_time'] = MergeSortRunTimeModel.simulate_rt(experience['problem'])
            else:
                experience['run_time'] = CocktailSortRunTimeModel.simulate_rt(experience['problem'])
            
            # for non-binary scores, base the score on the run-time, assuming lower is better, the score should be one of 1,2,3,4 or 5
            experience["score"] = max(1, min(5, 6 - int(experience['run_time'] / 1000)))  # example scoring function

            experience["score"] = int(experience['run_time'])  # example scoring function
            experiments[s]["learning_trials"][t] = experience
    
    test_trials = [
        {'problem': {'input': generator.generate_random_sequence(64, 1000), 'time_cost': 0.01}},
        {'problem': {'input': generator.generate_random_sequence(6, 1000), 'time_cost': 0.01}},
        {'problem': {'input': generator.generate_partially_ordered_sequence(64, 0.01), 'time_cost': 0.01}},
        {'problem': {'input': generator.generate_partially_ordered_sequence(6, 0.01), 'time_cost': 0.01}},
    ]
    return experiments, test_trials

def simulate_experiment(agent, learning_trials, test_trials):
    for _, exp in learning_trials.items():
        #print(exp['features'])
        exp['features'] = np.array(exp['features']).reshape(1,-1)
        #print(np.reshape(exp["problem"]["input"], (-1,1)))
        #exp['features2'] = agent.problem_analyzer.extract_features(np.reshape(exp["problem"]["input"], (-1,1)))
        #print(exp['features'])
        agent.reflect(exp)

    choices = []
    for test in test_trials:
        agent, solution, experience = agent.solve_problem(test['problem'], exp['features'])
        experience['problem'] = test['problem']
        #experience['feedback'] = int((solution == sorted(test['problem']['input'])).all())
        #experience['correct'] = experience['feedback']
        experience['time_cost'] = test['problem']['time_cost']
        choices.append(experience['algorithm'])
    return choices


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### Main script to run the sorting experiments


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_results = False

# Experiment setup
algorithm_names = ['merge sort', 'cocktail sort']
algorithm_params = np.array([0,0,0,0,1,1,1,1,0,1])
# the numbers in the arrays in feature_params epresent the number of elements to be sorted and the range of elements
# for example [4,4] means sorting 4 elements drawn from the range 1 to 4
feature_params = np.array([
    [4,4], [8,8], [16,16], [16,1], [4,4], [8,8], [16,16], [16,1], [36,36], [36,36]
])

# Hypotheses
sorting_algorithms = [merge_sort, cocktail_sort]  # Placeholders
nr_algorithms = len(sorting_algorithms)
parameters = {'betas': np.ones(nr_algorithms)/nr_algorithms, 'r_max': 1, 'w': 1}


model_based_agent = MetaCognitiveSortingAgent(False, sorting_algorithms)
categories = {
    'is_short': lambda input: len(input) <= 16,
    'is_long': lambda input: len(input) >= 32,
    'is_presorted': lambda input: np.mean(np.diff(input) >= 0) > 0.9,
    'is_disordered': lambda input: np.mean(np.diff(input) >= 0) < 0.5
}
SCADS_agent1 = SCADSSortingAgent(sorting_algorithms, 1, categories)
SCADS_agent2 = SCADSSortingAgent(sorting_algorithms, 2, categories)
SCADS_agent3 = SCADSSortingAgent(sorting_algorithms, 3, categories)

agents = [model_based_agent, SCADS_agent1, SCADS_agent2, SCADS_agent3]
model_names = ['VOC','SCADS1','SCADS2','SCADS3']

generator = SortingProblemGenerator()

nr_subjects = 4
nr_trials = len(feature_params)
# Learning trials

experiments, test_trials = create_experiments(nr_subjects,nr_trials)

#for t in experiments[0]['learning_trials']: print(experiments[0]['learning_trials'][t])
    
choices = np.zeros((len(test_trials), len(agents), nr_subjects), dtype=int)
for s in range(nr_subjects):
    for a, agent in enumerate(agents):
        choices[:,a,s] = simulate_experiment(agent, experiments[s]['learning_trials'], test_trials)

percentage_use_of_strategy2 = 100 * np.mean(choices == 2, axis=2)
std_strategy_use = np.sqrt(percentage_use_of_strategy2/100 * (1 - percentage_use_of_strategy2/100))
SEM_strategy_use = std_strategy_use / np.sqrt(nr_subjects)


if plot_results:
    # Plotting
    plt.figure()
    plt.bar(np.arange(len(model_names)), percentage_use_of_strategy2[0], yerr=100*1.96*(SEM_strategy_use[0]+0.01))
    plt.ylabel('Choice of Merge Sort (%)', fontsize=18)
    plt.xlabel('Strategy Selection Model', fontsize=18)
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45, fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure()
    for i in range(len(model_names)):
        plt.plot(percentage_use_of_strategy2[:,i], label=model_names[i], linewidth=2.5)
    plt.xticks(np.arange(4), ['1','2','3','4'])
    plt.xlabel('Problem Type', fontsize=14)
    plt.ylabel('Choice of Merge Sort in %', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    adaptive_change = percentage_use_of_strategy2[0,:] - np.mean(percentage_use_of_strategy2[1:,:], axis=0)
    std_change = np.sqrt(np.abs(adaptive_change/100) * (1 - np.abs(adaptive_change/100)))
    SEM_change = std_change / np.sqrt(nr_subjects)

    plt.figure()
    plt.bar(np.arange(len(model_names)), adaptive_change, yerr=100*1.96*SEM_change)
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.ylabel('Adaptive Change in Strategy Use (in %)', fontsize=18)
    plt.xlabel('Strategy Selection Model', fontsize=18)
    plt.tight_layout()
    plt.show() 