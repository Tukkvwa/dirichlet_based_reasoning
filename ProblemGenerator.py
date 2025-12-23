class ProblemGenerator:
    def __init__(self, scorer, solver):
        self.scorer = scorer
        self.solver = solver

    def generate_input(self):
        # Placeholder for input generation logic
        return []

    def generate_problem(self, params, time_cost):
        problem = {}
        problem["object"] = self.generate_input(params)
        problem["scorer"]  = self.scorer
        problem["solution"] = self.solver(self, problem["object"])
        problem["time_cost"] = time_cost
        problem["params"] = params

        return problem