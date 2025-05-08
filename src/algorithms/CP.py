import time
from ortools.sat.python import cp_model

from src.algorithms.algorithm import Algorithm
from src.constraints import *
from src.segmentation import Segment


class CpSolver(Algorithm):
    def __init__(self, items, segments, constraints, N):
        """
        Args:
          items: dict mapping candidate item id -> score.
          segments: dict mapping segment id -> Segment.
          constraints: list of constraints
          N: number of recommendation positions.
        """
        self.items = items
        self.segments = segments
        self.constraints = constraints
        self.N = N

    def build_model(self, with_objective=True):
        model = cp_model.CpModel()
        positions = list(range(self.N))
        # Create binary decision variables: x[i, p] = 1 if candidate i is placed at position p.
        x = {}
        for i in self.items:
            for p in positions:
                x[i, p] = model.NewBoolVar(f"x_{i}_{p}")
        # Each candidate is used at most once.
        for i in self.items:
            model.Add(sum(x[i, p] for p in positions) <= 1)
        # Each recommendation position must be filled by exactly one candidate.
        for p in positions:
            model.Add(sum(x[i, p] for i in self.items) == 1)
        if with_objective:
            model.Maximize(sum(self.items[i] * x[i, p] for i in self.items for p in positions))
        K = 1  # dummy scaling factor
        already_recommended_items = []  # Assume none for now.
        # Plug in each constraint.
        for constraint in self.constraints:
            constraint.add_to_cp_model(model, x, self.items, self.segments, 0, positions, self.N, K, already_recommended_items)
        return model, x, positions

    def solve_optimal(self):
        model, x, positions = self.build_model(with_objective=True)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        solution = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for p in positions:
                for i in self.items:
                    if solver.Value(x[i, p]) == 1:
                        solution[p] = i
                        break
        total_score = sum(self.items[solution[p]] for p in solution) if solution else 0
        return solution, total_score

    def solve_first_feasible(self):
        model, x, positions = self.build_model(with_objective=False)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        solution = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for p in positions:
                for i in self.items:
                    if solver.Value(x[i, p]) == 1:
                        solution[p] = i
                        break
        total_score = sum(self.items[solution[p]] for p in solution) if solution else 0
        return solution, total_score

    def check_constraints(self, solution):
        # Check if the solution satisfies all constraints.
        for constraint in self.constraints:
            if not constraint.check_constraint(solution, self.items, self.segments):
                return False
        return True


class PermutationCpSolver:
    def __init__(self, items, segments, constraints, N):
        """
        Args:
          items: dict mapping candidate id (e.g. 'item1') -> score (float).
          segments: dict mapping segment id -> Segment.
                    (A Segment must support membership testing, e.g. via __contains__.)
          constraints: list of constraint objects (each with an add_to_permutation_cp_model method).
          N: number of recommendation positions.
        """
        self.items = items
        self.segments = segments
        self.constraints = constraints
        self.N = N

        # Build mapping from candidate id to an integer
        self.candidate_to_int = {}
        self.int_to_candidate = {}
        for idx, candidate in enumerate(items.keys()):
            self.candidate_to_int[candidate] = idx
            self.int_to_candidate[idx] = candidate
        self.num_candidates = len(items)

        # To work with integer scores, multiply by a scaling factor
        self.score_multiplier = 10
        self.scores = [int(round(items[self.int_to_candidate[i]] * self.score_multiplier))
                       for i in range(self.num_candidates)]

        # For each segment, create an indicator list of length num_candidates
        # indicator_list[j] is 1 if candidate j belongs to the segment, 0 otherwise
        self.segment_indicator = {}
        for seg_id, seg in self.segments.items():
            self.segment_indicator[seg_id] = [1 if self.int_to_candidate[i] in seg else 0
                                               for i in range(self.num_candidates)]

        # Build global_segments: mapping from segmentation_property -> dict of seg_id -> indicator list
        self.global_segments = {}
        for seg_id, seg in self.segments.items():
            prop = getattr(seg, "property", None)
            if prop is not None:
                if prop not in self.global_segments:
                    self.global_segments[prop] = {}
                self.global_segments[prop][seg_id] = self.segment_indicator[seg_id]

    def build_model(self, with_objective=True):
        model = cp_model.CpModel()

        # Decision variables: assign[p] is the candidate (as an int) chosen for position p
        assign = []
        for p in range(self.N):
            var = model.NewIntVar(0, self.num_candidates - 1, f"assign_{p}")
            assign.append(var)
        # Enforce that all chosen candidates are different
        model.AddAllDifferent(assign)

        # Objective: maximize the sum of candidate scores
        score_vars = []
        for p in range(self.N):
            score_var = model.NewIntVar(min(self.scores), max(self.scores), f"score_{p}")
            model.AddElement(assign[p], self.scores, score_var)
            score_vars.append(score_var)
        if with_objective:
            model.Maximize(sum(score_vars))

        # Prepare helper data for constraints.
        solver_data = {
            "candidate_to_int": self.candidate_to_int,
            "num_candidates": self.num_candidates,
            "segment_indicator": self.segment_indicator,
            "global_segments": self.global_segments
        }

        # Let each constraint add its own constraints.
        for constraint in self.constraints:
            constraint.add_to_permutation_cp_model(model, assign, solver_data, self.N)
        return model, assign

    def solve_optimal(self):
        model, assign = self.build_model(with_objective=True)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        solution = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for p, var in enumerate(assign):
                candidate_int = solver.Value(var)
                solution[p] = self.int_to_candidate[candidate_int]
            total_score = sum(self.items[solution[p]] for p in solution)
        else:
            total_score = 0
        return solution, total_score

    def solve_first_feasible(self):
        model, assign = self.build_model(with_objective=False)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        solution = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for p, var in enumerate(assign):
                candidate_int = solver.Value(var)
                solution[p] = self.int_to_candidate[candidate_int]
            total_score = sum(self.items[solution[p]] for p in solution)
        else:
            total_score = 0
        return solution, total_score


###############################################
# Example Usage
###############################################

if __name__ == "__main__":
    items = {
        'item1': 9.0,
        'item2': 8.5,
        'item3': 8.0,
        'item4': 7.5,
        'item5': 7.0,
        'item6': 6.5,
        'item7': 6.0,
        'item8': 5.5,
        'item9': 5.0,
        'item10': 4.5
    }

    seg1 = Segment('genre1', 'genre', 'item1', 'item2', 'item3', 'item4', 'item5')
    seg2 = Segment('genre2', 'genre', 'item6', 'item7', 'item8', 'item9', 'item10')
    segments = {seg1.id: seg1, seg2.id: seg2}

    constraints = [
        # MinItemsPerSegmentConstraint(segment_id='genre1', min_items=1, window_size=5),
        # MaxItemsPerSegmentConstraint(segment_id='genre2', max_items=2, window_size=5),

        # GlobalMinItemsPerSegmentConstraint(segmentation_property='genre', min_items=1, window_size=5),
        # GlobalMaxItemsPerSegmentConstraint(segmentation_property='genre', max_items=3, window_size=5),

        MinSegmentsConstraint(segmentation_property='genre', min_segments=1, window_size=3),
        MaxSegmentsConstraint(segmentation_property='genre', max_segments=1, window_size=3)
    ]

    N = 5
    cp_solver = CpSolver(items, segments, constraints, N)

    # solve for the optimal solution
    start_time = time.time()
    opt_solution, opt_score = cp_solver.solve_optimal()
    solution_valid = cp_solver.check_constraints(opt_solution)
    print("Optimal solution (position -> candidate):", opt_solution)
    print("Optimal total score:", opt_score)
    print(f"Optimal solution time: {(time.time() - start_time) * 1000:.2f} ms.")
    print("Optimal solution valid:", solution_valid)

    # solve for a first feasible solution
    start_time = time.time()
    feas_solution, feas_score = cp_solver.solve_first_feasible()
    solution_valid = cp_solver.check_constraints(feas_solution)
    print("First feasible solution (position -> candidate):", feas_solution)
    print("Feasible total score:", feas_score)
    print(f"First feasible solution time: {(time.time() - start_time) * 1000:.2f} milliseconds.")
    print("Feasible solution valid:", solution_valid)


    # example usage of PermutationCpSolver
    solver = PermutationCpSolver(items, segments, constraints, N)

    # Solve for the optimal solution.
    start_time = time.time()
    opt_solution, opt_score = solver.solve_optimal()
    print("Optimal solution (position -> candidate):", opt_solution)
    print("Optimal total score:", opt_score)
    print(f"Optimal solution time: {(time.time() - start_time) * 1000:.2f} ms.")

    # Solve for a first feasible solution.
    start_time = time.time()
    feas_solution, feas_score = solver.solve_first_feasible()
    print("First feasible solution (position -> candidate):", feas_solution)
    print("Feasible total score:", feas_score)
    print(f"First feasible solution time: {(time.time() - start_time) * 1000:.2f} ms.")

