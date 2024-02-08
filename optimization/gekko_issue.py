from gekko import GEKKO

m = GEKKO()
s = m.Var(value=15.0, lb=10.0, ub=20.0, name="s")
c = m.Intermediate(-5.0, name="c")
a = m.Intermediate(c, name="a")
r = m.Intermediate(200000.0 * a ** 0.5, name="r")  # complex number
obj_expr = r - s
m.Maximize(obj_expr)
m.options.MAX_ITER = 100000
m.solve()
print("Done")
