import random
import os
import time
import argparse
import gc
import sys
import json
from datetime import datetime
from collections import Counter, defaultdict

# Color codes for terminal output
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'
BOLD_GREEN = '\033[1;32m'
BOLD_RED = '\033[1;31m'

def generate_3sat_cnf(num_vars, num_clauses, seed=None):
    # Generate 3-SAT CNF with specified variables and clauses
    if seed is not None:
        random.seed(seed)
    clauses = []
    for _ in range(num_clauses):
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, num_vars)
            lit = var if random.choice([True, False]) else -var
            clause.add(lit)
        clauses.append(list(clause))
    return clauses

def write_dimacs_file(filename, num_vars, clauses):
    # Write CNF to DIMACS file
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def generate_multiple_files(folder, num_files=100, num_vars=6, num_clauses=30, base_seed=None):
    # Generate multiple CNF files
    os.makedirs(folder, exist_ok=True)
    for i in range(num_files):
        seed = base_seed + i if base_seed is not None else i
        clauses = generate_3sat_cnf(num_vars, num_clauses, seed=seed)
        filename = os.path.join(folder, f"cnf_{num_vars}v_{num_clauses}c_{i+1}.cnf")
        write_dimacs_file(filename, num_vars, clauses)
        print(f"{GREEN}Generated: {filename} ✅{RESET}")

def parse_dimacs_cnf(file_path):
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
                print(f"{YELLOW}Warning: Invalid clause line: {line}{RESET}")
    return n_vars, n_clauses, clause_list

def propagate_units(clauses, indent="", verbose=True):
    # Simplify clauses via unit propagation
    units = [c for c in clauses if len(c) == 1]
    propagation_count = 0
    while units:
        unit = units.pop()
        literal = next(iter(unit))
        if verbose:
            print(f"{indent}{GREEN}Found unit literal {literal} 🚀{RESET}")
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
                        print(f"{indent}{RED}Conflict: empty clause after removing {-literal} ❌{RESET}")
                    return False, propagation_count
                if verbose:
                    print(f"{indent}{YELLOW}Removed {-literal}, new clause: {set(new_clause)}{RESET}")
                if len(new_clause) == 1:
                    units.append(new_clause)
                updated_clauses.append(new_clause)
            else:
                updated_clauses.append(clause)
        clauses = updated_clauses
    return clauses, propagation_count

def eliminate_pure_literals(clauses, indent="", verbose=True):
    # Remove clauses with pure literals
    literals = set(l for clause in clauses for l in clause)
    pure_literals = {l for l in literals if -l not in literals}
    if not pure_literals:
        return clauses
    filtered_clauses = []
    for clause in clauses:
        if any(l in pure_literals for l in clause):
            if verbose:
                print(f"{indent}{BLUE}Removed clause {set(clause)} (pure literal) 🌟{RESET}")
            continue
        filtered_clauses.append(clause)
    return eliminate_pure_literals(filtered_clauses, indent, verbose)

def simulate_unit_propagation(clauses, literal, indent, verbose):
    # Simulate literal assignment for propagation count
    test_clauses = clauses + [{literal}]
    result, count = propagate_units(test_clauses, indent, verbose=False)
    if result is False:
        return 0
    if not result:
        return float("inf")
    return count

def choose_literal(clauses, indent, verbose=True, strategy="first"):
    # Select branching literal using heuristic
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
    # Iterative DPLL algorithm for SAT solving
    indent = "  " * depth
    clauses, _ = propagate_units(clauses, indent, verbose)
    if clauses is False:
        if verbose:
            print(f"{indent}{RED}Unsatisfiable after unit propagation ❌{RESET}")
        return False, branch_count
    if not clauses:
        if verbose:
            print(f"{indent}{GREEN}Satisfiable after unit propagation ✅{RESET}")
        return True, branch_count
    clauses = eliminate_pure_literals(clauses, indent, verbose)
    if not clauses:
        if verbose:
            print(f"{indent}{GREEN}Satisfiable after pure literal elimination ✅{RESET}")
        return True, branch_count
    literal = choose_literal(clauses, indent, verbose, strategy)
    if literal is None:
        if verbose:
            print(f"{indent}{RED}No literals to branch on ❌{RESET}")
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
        print(f"{indent}{RED}Unsatisfiable: both branches failed ❌{RESET}")
    return False, branch_count

def resolve_pair(clause1, clause2):
    # Resolve two clauses
    for l in clause1:
        if -l in clause2:
            return [(clause1 - {l}) | (clause2 - {-l})]
    return []

def is_tautological(clause):
    # Check if clause is tautological
    return any(-l in clause for l in clause)

def resolution_core(clauses, preprocess=True, verbose=True):
    # Core resolution logic
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
                print(f"{GREEN}SATISFIABLE ✅{RESET}")
            return True
        clauses.extend(new)

def resolution_algorithm(clauses, verbose=True):
    # Classic resolution algorithm
    return resolution_core(clauses, preprocess=False, verbose=verbose)

def davis_putnam(clauses, verbose=True):
    # Davis-Putnam with preprocessing
    return resolution_core(clauses, preprocess=True, verbose=verbose)

def solve(clauses, method="first", verbose=False):
    # Solve SAT with selected method
    dpll_strategies = ["dpll", "first", "random", "MAXO", "MOMS", "MAMS", "JW", "UP", "GUP", "SUP"]
    if method in dpll_strategies:
        strategy = "first" if method == "dpll" else method
        result, splits = dpll_solver(clauses, strategy=strategy, verbose=verbose)
        return result, splits
    elif method == "resolution":
        return resolution_algorithm(clauses, verbose=verbose), 0
    elif method == "dp":
        return davis_putnam(clauses, verbose=verbose), 0
    else:
        raise ValueError(f"Unknown method: {method}")

def solve_cnf_file(file_path, method="first", verbose=False):
    # Solve a single CNF file
    num_vars, num_clauses, clauses = parse_dimacs_cnf(file_path)
    start_time = time.perf_counter()
    gc.disable()
    try:
        result, splits = solve(clauses, method, verbose=verbose)
    finally:
        gc.enable()
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, splits, result

def test_folder(folder_path, method="first"):
    # Test all CNF files in folder
    times = []
    splits_data = []
    satisfiable_count = 0
    total_files = 0
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".cnf"))
    if not files:
        raise FileNotFoundError(f"{RED}No CNF files in {folder_path} ❌{RESET}")
    for idx, file_name in enumerate(files, 1):
        total_files += 1
        print(f"{YELLOW}Testing [{idx}/{len(files)}] {file_name} with {method} 🔍{RESET}")
        file_path = os.path.join(folder_path, file_name)
        try:
            elapsed_time, splits, result = solve_cnf_file(file_path, method, verbose=False)
            times.append(elapsed_time)
            splits_data.append(splits)
            satisfiable_count += result
        except Exception as e:
            print(f"{RED}Error in {file_name}: {str(e)} ❌{RESET}")
    return satisfiable_count, total_files, times, splits_data

def benchmark_methods(folder_path, methods):
    # Benchmark multiple methods
    results = {}
    for method in methods:
        print(f"\n{BOLD}Benchmarking {method} 🔧{RESET}")
        satisfiable_count, total_files, times, splits_data = test_folder(folder_path, method)
        avg_time = sum(times) / len(times) if times else 0
        avg_splits = sum(splits_data) / len(splits_data) if splits_data else 0
        print(f"Method: {method}")
        print(f"Satisfiable: {satisfiable_count}/{total_files} ✅")
        print(f"Average time: {avg_time:.4f} seconds ⏱")
        print(f"Average splits: {avg_splits:.2f} 🔀")
        results[method] = {
            "satisfiable": satisfiable_count,
            "total": total_files,
            "times": times,
            "splits": splits_data,
            "avg_time": avg_time,
            "avg_splits": avg_splits
        }
    return results

def save_results_to_json(results, folder_path, filename="benchmark_results.json"):
    # Save benchmark results to JSON file
    full_path = os.path.join(folder_path, filename)
    with open(full_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"{GREEN}Saved results to {full_path} 📝{RESET}")

def validate_args(args):
    # Validate command-line arguments
    if not (args.generate or args.benchmark):
        print(f"{BOLD_RED}Error: Specify --generate or --benchmark ❌{RESET}")
        parser.print_help()
        sys.exit(1)
    if (args.generate or args.benchmark) and not args.folder:
        print(f"{BOLD_RED}Error: --folder required ❌{RESET}")
        sys.exit(1)
    if args.benchmark and not os.path.exists(args.folder):
        print(f"{BOLD_RED}Error: Folder {args.folder} not found ❌{RESET}")
        sys.exit(1)
    if args.generate and (args.num_vars < 3 or args.num_clauses < 3):
        print(f"{BOLD_RED}Error: Minimum 3 variables/clauses required ❌{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and benchmark SAT solvers.")
    parser.add_argument("--generate", action="store_true", help="Generate CNF files")
    parser.add_argument("--num_vars", type=int, default=6, help="Number of variables")
    parser.add_argument("--num_clauses", type=int, default=30, help="Number of clauses")
    parser.add_argument("--num_files", type=int, default=100, help="Number of files")
    parser.add_argument("--seed", type=int, help="Base seed for generation")
    parser.add_argument("--folder", type=str, default="cnf/generated", help="Folder for files")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=["dpll", "first", "random", "MAXO", "MOMS", "MAMS", "JW", "UP", "GUP", "SUP", "resolution", "dp"],
                        help="Methods to benchmark")
    args = parser.parse_args()
    validate_args(args)
    if args.generate:
        print(f"{BLUE}Generating {args.num_files} files with {args.num_vars} vars and {args.num_clauses} clauses 📄{RESET}")
        generate_multiple_files(args.folder, args.num_files, args.num_vars, args.num_clauses, args.seed)
    if args.benchmark:
        print(f"{BLUE}Benchmarking methods {args.methods} on {args.folder} 📈{RESET}")
        results = benchmark_methods(args.folder, args.methods)
        save_results_to_json(results, args.folder)