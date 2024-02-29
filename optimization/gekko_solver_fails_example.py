from gekko import GEKKO

m = GEKKO(remote=False)

s = m.Var(value=100.0, lb=0.0, name="s")

# c = m.Intermediate(285.0586650864552 * np.prod([np.power(s, 0.5533413522107526)]), name="c")
c = m.Intermediate(300.0 * s ** 0.553, name="c")

a = m.Intermediate(c + 0.0, name="a")

# r = m.Intermediate(231968.41453018063 * np.prod([np.power((a + 1.0), 0.00979506910307301) for i in range(1)]),
#                    name="r_0_t1")
r = m.Intermediate(230000.0 * (a + 1.0) ** 0.00979, name="r")

# spend_expr = m.Intermediate(m.sum([s]), name='spend')
spend_expr = m.Intermediate(s, name='spend')

revenue_expr = m.Intermediate(m.sum([r]), name='revenue')
# revenue_expr = m.Intermediate(r, name='revenue')

profit_expr = m.Intermediate(1.0 * revenue_expr - spend_expr, name='profit')

m.Maximize(profit_expr)

# m.options.SOLVER = 1  # APOPT (v1.0) -- fails with LB = 0, succeeds with LB > 0 (regardless of remote)
# m.options.SOLVER = 2  # BPOPT (v1.0) -- succeeds always regardless of remote
m.options.SOLVER = 3  # IPOPT (v3.12) -- succeeds with remote = True, fails with remote = False
# Solver 4 ==> "Invalid solver license: Please select another solver"

m.options.MAX_ITER = 100000
m.options.MAX_MEMORY = 10 ** 6
m.options.DIAGLEVEL = 2

m.solve()
print(s.VALUE[0])
print(profit_expr.VALUE[0])
