using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/EquationsToWaveform")

# Regenerate training data with current prob
fresh_sol = solve(prob, Tsit5(), saveat=0.1)
h_plus_training = compute_waveform(0.1, fresh_sol, 1.0)[1]
h_cross_training = compute_waveform(0.1, fresh_sol, 1.0)[2]

p_true = [1.0, 
          Elliptical_Orbit_Energy(R, M, BH_Kick, Angular_Momentum(R, M)), 
          Angular_Momentum(R, M)]

p_guess = [1.0, 
           0.5, 
           3.77]

function loss(pn, trainingFraction)
    newprob = remake(prob, p = pn)
    sol = solve(newprob, Tsit5(), saveat=0.1)
    predicted_waveform_plus = compute_waveform(0.1, sol, 1.0)[1]
    predicted_waveform_cross = compute_waveform(0.1, sol, 1.0)[2]
    
    # Compare only the overlapping portion
    n_train = Int(floor(length(h_plus_training)*trainingFraction))
    n_pred = length(predicted_waveform_plus)
    n_compare = min(n_pred, n_train)
    
    loss = sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
    loss += sum(abs2, predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare])
    println("Training with fraction: ", trainingFraction, ", n_compare: ", n_compare, ", loss: ", loss)
    return loss
end

trainingFraction = 1

pred_prob = remake(prob, p = p_guess)
sol_prob = solve(pred_prob, Tsit5(), saveat=0.1)
predicted_waveform_plus_old = compute_waveform(0.1, sol_prob, 1.0)[1]

println("True parameters: ", p_true)
println("The loss for the initial guess is ", loss(p_guess, 1))
println("The loss for the truth is ", loss(p_true, 1))

losses = []

function callback(pn, loss; dotrain = true)
    if dotrain
        push!(losses, loss);
        @printf("Epoch: %d, Loss: %15.12f \n",length(losses),loss);
        p = plot(losses, label = "Loss", xlabel = "Epochs", ylabel = "Loss", title = "Training Loss")
        display(p)
    else
        prinln(l)
    end
    return false
end

# define Optimization Problem
adtype = Optimization.AutoFiniteDiff() # instead of Optimization.AutoZygote(), use finite differences

optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)

# create Optimization Problem (function + initial guess for parameters)
optprob = Optimization.OptimizationProblem(optf, p_guess)

# choose method for solving problem 
lr = 1e-2;
opt_method = Optimisers.Adam(lr)

# solve the problem
num_iters = 60;

opt_result = Optimization.solve(optprob, opt_method, callback=callback, maxiters=num_iters)

p_final = opt_result.u

newprob = remake(prob, p = p_final)
# timestep = 1e-1
# t_fine = 0:timestep:200
sol = solve(newprob, Tsit5(), saveat=0.1)

h_plus_pred = compute_waveform(0.1, sol, 1.0)[1]
h_cross_pred = compute_waveform(0.1, sol, 1.0)[2]

# Handle all cases properly
n_pred = length(h_plus_pred)
n_train = length(h_plus_training)

if n_pred == n_train
    # Same length - no padding needed
    h_plus_pred_plot = h_plus_pred
    h_plus_train_plot = h_plus_training
    n_plot = n_train
elseif n_pred < n_train
    # Predicted is shorter - pad with zeros
    h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
    h_plus_train_plot = h_plus_training
    n_plot = n_train
else
    # Predicted is longer - truncate to match training
    h_plus_pred_plot = h_plus_pred[1:n_train]
    h_plus_train_plot = h_plus_training
    n_plot = n_train
end

t_plot = (0:n_plot-1) * 0.1
differences = h_plus_pred_padded .- h_plus_training
sorted_diffs = sort(differences, rev=true)

plot(t_plot, h_plus_train_plot, label="h+ true", linewidth=2,
     xlabel="Time (s)", ylabel="h+ Amplitude",
     legend=:topright, grid=true, top_margin = 10mm, left_margin = 10mm)
plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)# plot!(t_plot, predicted_waveform_plus_old_padded, label="h+ initial prediction", linewidth=2)