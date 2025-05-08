import time
from ortools.sat.python import cp_model

from src.algorithms.algorithm import Algorithm
from src.constraints import *
from src.segmentation import Segment



class CpSolver(Algorithm):

    def __init__(
        self,
        name: str = "CP‑SAT",
        description: str = "Constraint Programming Solver (OR‑Tools)",
        verbose: bool = True,
        time_limit: float | None = None,  # seconds, *None* means unlimited
    ) -> None:
        super().__init__(name, description, verbose)
        self.time_limit = time_limit

    def solve(
        self,
        items: Dict[str, float],
        segments: Dict[str, Segment],
        constraints: List[Constraint],
        N: int,
        already_recommended_items: List[str] | None = None,
        return_first_feasible: bool = False,
        num_threads: int = 0,
    ) -> Dict[int, str] | None:

        if self.verbose:
            print(
                f"[{self.name}] Solving CP‑SAT with {len(items)} items, {len(segments)} segments, "
                f"{len(constraints)} constraints, N={N}."
            )

        if already_recommended_items is None:
            already_recommended_items = []

        # Build CP model
        with_objective = not return_first_feasible
        model, x, positions = self._build_model(
            items,
            segments,
            constraints,
            N,
            already_recommended_items,
            with_objective=with_objective,
        )

        # Configure solver
        solver = cp_model.CpSolver()
        # Logging & limits
        solver.parameters.log_search_progress = self.verbose
        if self.time_limit is not None:
            solver.parameters.max_time_in_seconds = self.time_limit
        if num_threads > 0:
            solver.parameters.num_search_workers = num_threads
        if return_first_feasible:
            # Stop as soon as a feasible solution is found
            solver.parameters.stop_after_first_solution = True

        # Search
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if self.verbose:
                print(f"[{self.name}] No feasible solution found (status={status}).")
            return None

        # Extract solution
        solution: Dict[int, str] = {}
        for p in positions:
            for i in items:
                if solver.Value(x[i, p]):
                    solution[p] = i
                    break

        if self.verbose:
            objective = solver.ObjectiveValue() if with_objective else "-"
            status_name = {
                cp_model.OPTIMAL: "OPTIMAL",
                cp_model.FEASIBLE: "FEASIBLE",
            }[status]
            print(
                f"[{self.name}] {status_name} solution with objective={objective} "
                f"found in {solver.WallTime()*1000:.2f}ms."
            )

        # Return slate ordered by position (1...N)
        return {pos: solution[pos] for pos in sorted(solution)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_model(
        self,
        items: Dict[str, float],
        segments: Dict[str, Segment],
        constraints: List[Constraint],
        N: int,
        already_recommended_items: List[str],
        *,
        with_objective: bool = True,
    ) -> tuple[cp_model.CpModel, dict[tuple[str, int], cp_model.IntVar], List[int]]:
        """Construct the CP‑SAT model.

        Returns
        -------
        model, x, positions
            The OR‑Tools model, variable dictionary, and list of positions.
        """
        model = cp_model.CpModel()
        positions = list(range(1, N + 1))

        # Decision variables: x[i, p] == 1 ⇔ item *i* placed at position *p*
        x: dict[tuple[str, int], cp_model.IntVar] = {
            (i, p): model.NewBoolVar(f"x_{i}_{p}") for i in items for p in positions
        }

        # Each item can appear at most once
        for i in items:
            model.Add(sum(x[i, p] for p in positions) <= 1)

        # Each position must be filled by exactly one item
        for p in positions:
            model.Add(sum(x[i, p] for i in items) == 1)

        # Objective: maximise engagement score (unless caller disables it)
        if with_objective:
            model.Maximize(
                sum(items[i] * x[i, p] for i in items for p in positions)
            )

        # Penalty scaling factor – max theoretical score for top‑N items
        K = sum(sorted(items.values(), reverse=True)[:N])

        # Delegate additional business constraints
        for constraint in constraints:
            constraint.add_to_cp_model(
                model,
                x,
                items,
                segments,
                0,  # row index (for compatibility with 2‑D slates)
                positions,
                N,
                K,
                already_recommended_items,
            )

        return model, x, positions


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

