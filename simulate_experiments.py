import numpy as np
import matplotlib.pyplot as plt
from CocktailSortRTModel import CocktailSortRunTimeModel
from MergeSortRTModel import MergeSortRunTimeModel
from Sorter import merge_sort, cocktail_sort
from SortingProblemGenerator import SortingProblemGenerator
from MetaCognitiveSortingAgent import MetaCognitiveSortingAgent
from SCADSSortingAgent import SCADSSortingAgent
# Placeholders for agent classes and generator

SEED = 9

def create_experiments(train_trial_params, algorithms, algorithmsRT):
    """
    Create learning trials for nr_subjects number of subjects.
    Returns:
        experiments: dict of experiments for each subject
        test_trials: list of test trials
    """
    experiments = []
    for problem_params in train_trial_params:
        expmnt = {}
        expmnt["duration"] = 0 #duration of learning tials, this is a placeholder
        expmnt["learning_trials"] = []
        for no_alg, algorithm in enumerate(algorithms):
            train_trial = {
                'algorithm': no_alg,
                'problem_params': problem_params,
                'problem': generator.generate_problem(problem_params, 0.01),
            }
            train_trial['run_time'] = algorithmsRT[no_alg].simulate_rt(train_trial['problem']['object'], seed=SEED)    

            # for non-binary scores, base the score on the run-time, assuming lower is better, the score should be one of 1,2,3,4 or 5
            #train_trial["score"] = max(1, min(5, 6 - int(train_trial['run_time'] / 1000)))  # example scoring function
            train_trial["score"] = int(train_trial['run_time'])  # example scoring function
            expmnt["learning_trials"].append(train_trial)
        experiments.append(expmnt)
    
    test_trials = [
        generator.generate_problem([64, 100], 0.01),
        generator.generate_problem([6, 100], 0.01),
        generator.generate_problem([64, 100], 0.01, "pos"),
        generator.generate_problem([6, 100], 0.01, "pos")
    ]
    return experiments, test_trials

def simulate_experiment(agent, experiments, test_trials):
    choices = []
    for expmt in experiments:
        for train in expmt["learning_trials"]:
            train['features'] = agent.problem_analyzer.extract_features(train["problem"]["object"])
            agent.reflect(train)

    for test in test_trials:
        print(test["object"])
        test_features = agent.problem_analyzer.extract_features(test["object"])
        agent, choice, solution = agent.solve_problem(test, test_features)
        #experience['feedback'] = int((solution == sorted(test['problem']['input'])).all())
        #experience['correct'] = experience['feedback']
        choices.append(choice)
    return choices


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### Main script to run the sorting experiments


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# the numbers in the arrays in train_trial_params represent the number of elements to be sorted and the range of elements
# for example [4,4] means sorting 4 elements drawn from the range 1 to 4
train_trial_params = np.array([
    [4,4], [8,8], [16,16], [16,1], [4,4],
    [8,8], [16,16], [16,1], [36,36], [36,36]
])

# Hypotheses
algorithms = [merge_sort, cocktail_sort]
algorithmsRT = [MergeSortRunTimeModel, CocktailSortRunTimeModel]  # Placeholders
algorithm_params = np.array([0,0,0,0,1,1,1,1,0,1]) #which algorithm to use for each of the 10 trials, 0 = merge sort, 1 = cocktail sort

nr_algorithms = len(algorithms)
#parameters = {'betas': np.ones(nr_algorithms)/nr_algorithms, 'r_max': 1, 'w': 1}


model_based_agent = MetaCognitiveSortingAgent(False, algorithms, seed=SEED)
categories = {
    'is_short': lambda input: len(input) <= 16,
    'is_long': lambda input: len(input) >= 32,
    'is_presorted': lambda input: np.mean(np.diff(input) >= 0) > 0.9,
    'is_disordered': lambda input: np.mean(np.diff(input) >= 0) < 0.5
}
SCADS_agent1 = SCADSSortingAgent(algorithms, 1, categories, seed=SEED)
SCADS_agent2 = SCADSSortingAgent(algorithms, 2, categories, seed=SEED)
SCADS_agent3 = SCADSSortingAgent(algorithms, 3, categories, seed=SEED)

agents = [model_based_agent, SCADS_agent1, SCADS_agent2, SCADS_agent3]
model_names = ['VOC','SCADS1','SCADS2','SCADS3']

generator = SortingProblemGenerator(seed=SEED)

nr_subjects = len(agents)

# Creating training (in experiments' learning trials) and learning trials
experiments, test_trials = create_experiments(train_trial_params, algorithms, algorithmsRT)

#for t in experiments[0]['learning_trials']: print(experiments[0]['learning_trials'][t])
choices = []

for a, agent in enumerate(agents):
    choices.append(simulate_experiment(agent, experiments, test_trials))

print(choices)

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### Plotting


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


percentage_use_of_strategy2 = 100 * np.mean(choices, axis=1)
std_strategy_use = np.sqrt(percentage_use_of_strategy2/100 * (1 - percentage_use_of_strategy2/100))
SEM_strategy_use = std_strategy_use / np.sqrt(nr_subjects)

#plot_results = True
plot_results = False

if plot_results:
    # Plotting
    plt.figure()
    plt.bar(np.arange(len(model_names)), percentage_use_of_strategy2, yerr=100*1.96*(SEM_strategy_use+0.01))
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