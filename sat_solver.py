import argparse
import random
from collections import Counter

# Color codes for terminal output
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'
BOLD_GREEN = '\033[1;32m'
BOLD_RED = '\033[1;31m'

def propagate_units(clauses, indent="", verbose=True):
    # Unit propagation to simplify clauses
    units = [c for c in clauses if len(c) == 1]
    propagation_count = 0
    while units:
        unit = units.pop()
        literal = next(iter(unit))
        if verbose:
            print(f"{indent}{GREEN}Found unit literal {literal}{RESET}")
        updated_clauses = []
        for clause in clauses:
            if literal in clause:
                if verbose:
                    print(f"{indent}{BLUE}Removed clause {set(clause)}{RESET}")
                continue
            if -literal in clause:
                new_clause = clause - {-literal}
                propagation_count += 1
                if not new_clause:
                    if verbose:
                        print(f"{indent}{RED}Conflict: removed {-literal} from {set(clause)}, empty clause remains{RESET} ❌")
                    return False, propagation_count
                if verbose:
                    print(f"{indent}{YELLOW}Removed {-literal} from {set(clause)}, resulting in {set(new_clause)}{RESET}")
                if len(new_clause) == 1:
                    units.append(new_clause)
                updated_clauses.append(new_clause)
            else:
                updated_clauses.append(clause)
        clauses = updated_clauses
    return clauses, propagation_count

def eliminate_pure_literals(clauses, indent="", verbose=True):
    # Eliminate all clauses containing pure literals
    literals = set(l for clause in clauses for l in clause)
    pure_literals = {l for l in literals if -l not in literals}
    if not pure_literals:
        return clauses
    filtered_clauses = []
    for clause in clauses:
        if any(l in pure_literals for l in clause):
            if verbose:
                print(f"{indent}{BLUE}Removed clause {set(clause)} (pure literal present){RESET}")
            continue
        filtered_clauses.append(clause)
    return eliminate_pure_literals(filtered_clauses, indent, verbose)

def simulate_unit_propagation(clauses, literal, indent, verbose):
    # Simulate assigning a literal and count resulting unit propagations
    test_clauses = clauses + [{literal}]
    result, count = propagate_units(test_clauses, indent, verbose=False)
    if result is False:
        return 0
    if not result:
        return float("inf")
    return count

def choose_literal(clauses, indent, verbose=True, strategy="first"):
    # Select a literal for branching using the specified heuristic
    literals = [l for clause in clauses for l in clause]
    literal_set = set(literals)
    if not literal_set:
        return None

    if strategy == "first":
        return next(iter(next(iter(clauses))))
    if strategy == "random":
        return random.choice(list(literal_set))
    if strategy == "MAXO":
        return Counter(literals).most_common(1)[0][0]
    if strategy == "MOMS":
        min_len = min(len(c) for c in clauses)
        min_clauses = [c for c in clauses if len(c) == min_len]
        return Counter(l for c in min_clauses for l in c).most_common(1)[0][0]
    if strategy == "MAMS":
        overall = Counter(literals)
        min_len = min(len(c) for c in clauses)
        min_clauses = [c for c in clauses if len(c) == min_len]
        min_counter = Counter(l for c in min_clauses for l in c)
        best, best_score = None, -1
        for l in literal_set:
            score = overall[l] + min_counter[-l]
            if score > best_score:
                best_score, best = score, l
        return best
    if strategy == "JW":
        scores = {}
        for clause in clauses:
            for l in clause:
                scores[l] = scores.get(l, 0) + 2 ** (-len(clause))
        return max(scores, key=scores.get)
    if strategy in ["UP", "GUP"]:
        best, best_score = None, -1
        for l in literal_set:
            score = simulate_unit_propagation(clauses, l, indent, verbose)
            if strategy == "GUP" and (score == 0 or score == float('inf')):
                return l
            if score > best_score:
                best_score, best = score, l
        return best
    if strategy == "SUP":
        candidates = list(set([choose_literal(clauses, indent, verbose, m)
                               for m in ["MAXO", "MOMS", "MAMS", "JW"]]))
        best, best_score = None, -1
        for l in candidates:
            score = simulate_unit_propagation(clauses, l, indent, verbose)
            if score > best_score:
                best_score, best = score, l
        return best
    raise ValueError(f"Unknown strategy: {strategy}")

def dpll_solver(clauses, branch=None, depth=0, verbose=True, strategy="first", branch_count=0):
    # DPLL algorithm with advanced heuristics
    indent = "  " * depth
    clauses, _ = propagate_units(clauses, indent, verbose)
    if clauses is False:
        if verbose:
            print(f"{indent}{RED}Unsatisfiable after unit propagation{RESET} ❌" + (f" [branch {branch}]" if branch else ""))
        return False, branch_count
    if not clauses:
        if verbose:
            print(f"{indent}{GREEN}Satisfiable after unit propagation{RESET} ✅" + (f" [branch {branch}]" if branch else ""))
        return True, branch_count

    clauses = eliminate_pure_literals(clauses, indent, verbose)
    if not clauses:
        if verbose:
            print(f"{indent}{GREEN}Satisfiable after pure literal elimination{RESET} ✅" + (f" [branch {branch}]" if branch else ""))
        return True, branch_count

    literal = choose_literal(clauses, indent, verbose, strategy)
    if literal is None:
        if verbose:
            print(f"{indent}{RED}No literal left to branch on, backtracking{RESET}")
        return False, branch_count
    branch_count += 1

    if verbose:
        print(f"\n{indent}🔀 Branching on {literal} = True")
    result, branch_count = dpll_solver(clauses + [{literal}], literal, depth + 1, verbose, strategy, branch_count)
    if result:
        return True, branch_count

    if verbose:
        print(f"\n{indent}🔀 Branching on {literal} = False")
    result, branch_count = dpll_solver(clauses + [{-literal}], -literal, depth + 1, verbose, strategy, branch_count)
    if result:
        return True, branch_count

    if verbose:
        print(f"{indent}{RED}Unsatisfiable: both branches failed for {literal}{RESET} ❌")
    return False, branch_count

def resolve_pair(clause1, clause2):
    # Resolve two clauses if possible
    for l in clause1:
        if -l in clause2:
            return [(clause1 - {l}) | (clause2 - {-l})]
    return []

def is_tautological(clause):
    # Check if a clause is a tautology
    return any(-l in clause for l in clause)

def resolution_core(clauses, preprocess=True, verbose=True):
    # Resolution method core logic, optionally with DP preprocessing
    step = 1
    clauses = [frozenset(c) for c in clauses]
    if verbose:
        for c in clauses:
            print(f"({step}) {set(c)}")
            step += 1
        print()
    while True:
        new = set()
        if not clauses:
            if preprocess and verbose:
                print(f"{GREEN}SATISFIABLE ✅{RESET}")
            return True
        if preprocess:
            clauses, _ = propagate_units(clauses, verbose=verbose)
            if clauses is False:
                if verbose:
                    print(f"{RED}UNSATISFIABLE ❌{RESET}")
                return False
            if not clauses:
                if verbose:
                    print(f"{GREEN}SATISFIABLE ✅{RESET}")
                return True
            clauses = eliminate_pure_literals(clauses, verbose=verbose)
            if not clauses:
                if verbose:
                    print(f"{GREEN}SATISFIABLE ✅{RESET}")
                return True
        clause_set = set(clauses)
        pairs = [(clauses[i], clauses[j]) for i in range(len(clauses)) for j in range(i + 1, len(clauses))]
        for c1, c2 in pairs:
            resolvents = resolve_pair(c1, c2)
            for resolvent in resolvents:
                if not resolvent:
                    if verbose:
                        print(f"({step}) {RED}∅{RESET} from {set(c1)} and {set(c2)}")
                        print(f"{RED}UNSATISFIABLE ❌{RESET}")
                    return False
                if frozenset(resolvent) not in clause_set and not is_tautological(resolvent):
                    if verbose:
                        print(f"({step}) {BLUE}{set(resolvent)}{RESET} from {set(c1)} and {set(c2)}")
                    clause_set.add(frozenset(resolvent))
                    new.add(frozenset(resolvent))
                    step += 1
        if not new:
            if verbose:
                print("\nNo new resolvent can be added")
                print(f"{GREEN}SATISFIABLE ✅{RESET}")
            return True
        clauses.extend(new)

def resolution_algorithm(clauses, verbose=True):
    # Classic resolution algorithm
    return resolution_core(clauses, preprocess=False, verbose=verbose)

def davis_putnam(clauses, verbose=True):
    # Davis–Putnam (DP) method: resolution with preprocessing
    return resolution_core(clauses, preprocess=True, verbose=verbose)

def parse_dimacs(file_path):
    # Parse DIMACS CNF file
    n_vars, n_clauses, clause_list = 0, 0, []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(('c', '%', '0')):
                continue
            if line.startswith('p'):
                _, _, n_vars, n_clauses = line.split()
                n_vars, n_clauses = int(n_vars), int(n_clauses)
                continue
            try:
                clause = set(map(int, line.split()))
                if 0 in clause:
                    clause.remove(0)
                    clause_list.append(clause)
            except ValueError:
                print(f"{YELLOW}Warning: Invalid clause line format: {line}{RESET}")
    return n_vars, n_clauses, clause_list

def solve_formula(clauses, method="first", verbose=False):
    # Solve SAT with selected algorithm
    dpll_strategies = ["dpll", "first", "random", "MAXO", "MOMS", "MAMS", "JW", "UP", "GUP", "SUP"]
    if method in dpll_strategies:
        # Use "first" as the default strategy for "dpll" method
        strategy = "first" if method == "dpll" else method
        result, _ = dpll_solver(clauses, strategy=strategy, verbose=verbose)
        return result
    elif method == "resolution":
        return resolution_algorithm(clauses, verbose=verbose)
    elif method == "dp":
        return davis_putnam(clauses, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    # Main function to run the SAT solver
    parser = argparse.ArgumentParser(description="Enhanced SAT Solver with Multiple Algorithms including DPLL.")
    parser.add_argument("file", type=str, help="Path to CNF file.")
    parser.add_argument("--method", type=str, default="first",
                        help="Algorithm: resolution, dp, dpll, or DPLL branching strategies (first, random, MAXO, MOMS, MAMS, JW, UP, GUP, SUP)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()
    print(f"{BOLD}🔧 Starting SAT Solver with method: {args.method}{RESET}")
    try:
        n_vars, n_clauses, clauses = parse_dimacs(args.file)
    except FileNotFoundError:
        print(f"{RED}Error: File not found: {args.file}{RESET}")
        return
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        return
    result = solve_formula(clauses, method=args.method, verbose=args.verbose)
    if result:
        print(f"{BOLD_GREEN}SATISFIABLE ✅{RESET}")
    else:
        print(f"{BOLD_RED}UNSATISFIABLE ❌{RESET}")

if __name__ == "__main__":
    main()