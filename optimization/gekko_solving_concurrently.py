"""
Example that shows how to solve a problem with all 3 solvers in parallel using threading_and_processes.
Depending on the problem characteristics the performance of the solvers differs greatly.
Therefore, it is reasonable to try them all and see which one is fasted.
"""
import multiprocessing
import time
from multiprocessing.process import BaseProcess

import numpy as np
from gekko import GEKKO


# Note: this script is called every time a subprocess is created


def get_model(coefficients):
    n = len(coefficients)
    m = GEKKO(remote=False)
    variable_i = [m.Var(value=0, lb=0.0, ub=100.0, name=f"x_{i}") for i in range(n)]
    objective_expression = m.sum([coefficients[i] * variable_i[i] for i in range(n)])
    m.my_obj = objective_expression
    m.Maximize(objective_expression)
    m.Equation(m.sum(variable_i) <= 50.0)
    return m


def solve_apopt(coefficients, success: multiprocessing.Event = None, solutions=None):
    print("Calling APOPT ...")
    start = time.time()

    # Artificial delay
    time.sleep(10)

    m = get_model(coefficients)
    m.options.SOLVER = 1
    m.solve(disp=False)
    duration_sec = time.time() - start
    print(f"APOPT solved model in {duration_sec:1.2f} seconds. "
          f"Objective = {m.my_obj.VALUE[0]:4.2f}")
    if solutions is not None:
        solutions.put("AP")  # We could add an actual solution here
    if success is not None:
        success.set()


def solve_bpopt(coefficients, success: multiprocessing.Event = None, solutions=None):
    print("Calling BPOPT ...")
    start = time.time()

    # Artificial delay
    time.sleep(5)

    m = get_model(coefficients)
    m.options.SOLVER = 2
    m.solve(disp=False)
    duration_sec = time.time() - start
    print(f"BPOPT solved model in {duration_sec:1.2f} seconds. "
          f"Objective = {m.my_obj.VALUE[0]:4.2f}")
    if solutions is not None:
        solutions.put("BP")  # We could add an actual solution here
    if success is not None:
        success.set()


def solve_ipopt(coefficients, success: multiprocessing.Event = None, solutions=None):
    print("Calling IPOPT ...")
    start = time.time()

    # Small artificial delay => this one will be fastest
    time.sleep(1)

    m = get_model(coefficients)
    m.options.SOLVER = 3
    m.solve(disp=False)
    duration_sec = time.time() - start
    print(f"IPOPT solved model in {duration_sec:1.2f} seconds. "
          f"Objective = {m.my_obj.VALUE[0]:4.2f}")
    if solutions is not None:
        solutions.put("IP")  # We could add an actual solution here
    if success is not None:
        success.set()


def end_process(p: BaseProcess):
    if p.is_alive():  # => the thread is still active
        print(f"Process {p.name} still running => terminate it")
        p.terminate()  # may not work if process is stuck for good
        p.join()  # must be called after terminate or the process remains alive
        if p.is_alive():
            print("Process could not be terminated => kill it")
            p.kill()  # works for sure, no chance for process to finish nicely however
            p.join()
    assert not p.is_alive()


if __name__ == "__main__":
    # Create data
    n = 10
    coefficients = np.random.rand(n)

    # Create the processes to solve
    p1 = multiprocessing.Process(target=solve_apopt, args=(coefficients,))
    p2 = multiprocessing.Process(target=solve_bpopt, args=(coefficients,))
    p3 = multiprocessing.Process(target=solve_ipopt, args=(coefficients,))

    # Start processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for all processes to finish
    p1.join(15)
    p2.join(15)
    p3.join(15)
    assert not p1.is_alive()
    assert not p2.is_alive()
    assert not p3.is_alive()

    # Now we do the same thing, but use an event to terminate after we have received the 1st result
    print("\n\n\n")
    success = multiprocessing.Event()
    solutions = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=solve_apopt, args=(coefficients, success, solutions), name="AP")
    p2 = multiprocessing.Process(target=solve_bpopt, args=(coefficients, success, solutions), name="BP")
    p3 = multiprocessing.Process(target=solve_ipopt, args=(coefficients, success, solutions), name="IP")
    p1.start()
    p2.start()
    p3.start()
    success.wait()
    end_process(p1)
    end_process(p2)
    end_process(p3)
    while not solutions.empty():
        print(f"Solution: {solutions.get()}")
