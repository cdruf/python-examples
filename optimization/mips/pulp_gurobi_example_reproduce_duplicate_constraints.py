import pulp as pl

"""
The model has 2 constraints. However, the Gurobi output becomes 
Optimize a model with 4 rows, 2 columns and 6 nonzeros

This problem occurred with Gurobi 11 and PuLP 2.9.0.
Still occurs with Gurobi 12 and PuLP 3.1.1.
"""


def get_mdl():
    mdl = pl.LpProblem("MyLP", pl.LpMaximize)
    xs = pl.LpVariable.dicts('x', indices=range(2), cat=pl.LpInteger)
    mdl += pl.lpSum((i + 1) * x for i, x in xs.items())
    mdl += pl.lpSum(x for x in xs.values()) <= 5, "constraint_1"
    mdl += xs[1] <= 4, "constraint_2"
    mdl.writeMPS("my_example_mip.mps")
    mdl.writeLP("my_example_mip.lp")
    return mdl


if __name__ == "__main__":
    # Problematic variant
    print("### 1 ###")
    mdl1 = get_mdl()
    solver = pl.GUROBI(timeLimit=120, mipgap=0.0001)
    solver.buildSolverModel(mdl1)  # This creates the solverModel, but also the issue
    mdl1.solverModel.Params.IntFeasTol = 1e-9  # Use solverModel to set parameters

    mdl1.solve(solver)
    print(f"Objective value = {pl.value(mdl1.objective):.0f}")
    print("\n" * 5)

    # Normal variant does not allow setting custom Gurobi parameters
    print("### 2 ###")
    mdl2 = get_mdl()
    solver = pl.GUROBI(timeLimit=120, mipgap=0.0001)
    mdl2.solve(solver)
    print(f"Objective value = {pl.value(mdl2.objective):.0f}")
    print("\n" * 5)

    # Workaround (with custom parameters and start solution)
    print("### 3 ###")
    mdl3 = get_mdl()
    solver = pl.GUROBI(timeLimit=120, mipgap=0.0001, warmStart=True)  # Indicate warm start
    solver.initGurobi()
    solver.model.Params.IntFeasTol = 1e-9  # note: not using `solverModel`!
    # set optimal values for warm start
    mdl3.variables()[0].setInitialValue(1)
    mdl3.variables()[1].setInitialValue(4)

    # solver_var = self._mdl.solverModel.getVarByName(var.name)
    # val = round(var.value()) if is_int else var.value()
    # solver_var.setAttr('Start', val)

    mdl3.solve(solver)
    print(f"Objective value = {pl.value(mdl1.objective):.0f}")
