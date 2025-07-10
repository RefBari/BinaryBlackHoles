import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report
import Plots

# Dataset with two named features:
X = (a = rand(500), b = rand(500))

# and one target:
y = @. 2 * cos(X.a * 23.5) - X.b ^ 2

# with some noise:
y = y .+ randn(500) .* 1e-3

model = SRRegressor(
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[cos],
)

mach = machine(model, X, y)

fit!(mach)

report(mach)

predict(mach, X)