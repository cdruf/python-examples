import highspy
import numpy as np

# Create model
model = highspy.Highs()
coefficients = np.array([1.0])
lower_bounds = 0
upper_bounds = highspy.kHighsInf
model.addVars(coefficients, lower_bounds, upper_bounds)
model.changeObjectiveSense(highspy.ObjSense.kMaximize)

# Solve
model.run()

# Get solution
sol = model.getSolution()
print("Optimal Solution:", sol.col_value)
