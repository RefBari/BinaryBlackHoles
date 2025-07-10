u0 = Float32[pi,0.0,10,0]
tspan = (5.21, 6.78)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize)
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100
mass_ratio = 1.0
model_params = [mass_ratio]
mass1 = mass_ratio/(1.0+mass_ratio)
mass2 = 1.0/(1.0+mass_ratio)

x, y = file2trajectory(tsteps, "./input/trajectoryA.txt")
x2, y2 = file2trajectory()