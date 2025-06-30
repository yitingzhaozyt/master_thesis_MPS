using JuMP, Gurobi, Random, CSV, DataFrames, Plots

# Data
T = 12

Random.seed!(42)
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

demand_data = Dict{String, Vector{Int}}()
for prod in products
    demand_data[prod] = [rand(100:250) for t in 1:T]
end

production_caps = Dict("A" => 350, "B" => 350, "C" => 350)

product_resource_params = Dict(
    "product1" => Dict(
        "A" => Dict("prod_cost" => 6.62, "carbon_emission" => 10.562),
        "B" => Dict("prod_cost" => 8.73, "carbon_emission" => 9.695),
        "C" => Dict("prod_cost" => 12.74, "carbon_emission" => 7.699)
    ),
    "product2" => Dict(
        "A" => Dict("prod_cost" => 5.41, "carbon_emission" => 10.132),
        "B" => Dict("prod_cost" => 8.39, "carbon_emission" => 9.928),
        "C" => Dict("prod_cost" => 11.44, "carbon_emission" => 7.779)
    ),
    "product3" => Dict(
        "A" => Dict("prod_cost" => 6.85, "carbon_emission" => 11.583),
        "B" => Dict("prod_cost" => 8.15, "carbon_emission" => 9.392),
        "C" => Dict("prod_cost" => 10.4, "carbon_emission" => 7.623)
    ),
    "product4" => Dict(
        "A" => Dict("prod_cost" => 7.79, "carbon_emission" => 10.871),
        "B" => Dict("prod_cost" => 9.95, "carbon_emission" => 9.875),
        "C" => Dict("prod_cost" => 12.96, "carbon_emission" => 7.416)
    ),
    "product5" => Dict(
        "A" => Dict("prod_cost" => 7.08, "carbon_emission" => 11.175),
        "B" => Dict("prod_cost" => 8.7, "carbon_emission" => 8.173),
        "C" => Dict("prod_cost" => 12.81, "carbon_emission" => 7.558)
    )
)

inventory_costs = Dict("product1" => 1.6, "product2" => 2.06, "product3" => 2.67, "product4" => 1.93, "product5" => 1.23)

inventory_carbon = Dict("product1" => 1.172, "product2" => 0.507, "product3" => 1.69, "product4" => 1.924, "product5" => 0.904)

backorder_costs = Dict("product1" => 9.79, "product2" => 14.15, "product3" => 10.23, "product4" => 12.37, "product5" => 12.5)

waste_costs = Dict("product1" => 0.53, "product2" => 0.27, "product3" => 0.82, "product4" => 0.32, "product5" => 0.14)

spoilage_rates = Dict("product1" => 0.099, "product2" => 0.126, "product3" => 0.131, "product4" => 0.112, "product5" => 0.111)

cd(@__DIR__)

function deterministic_product_production(caps)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, s[products, 1:T] >= 0)
    @variable(model, b[products, 1:T] >= 0)
    @variable(model, x[products, resources, 1:T] >= 0)
    @variable(model, f[products, 1:T] >= 0)

    for prod in products
        @constraint(model, s[prod, 1] >= sum(x[prod, r, 1] for r in resources) - demand_data[prod][1])
        @constraint(model, b[prod, 1] >= -sum(x[prod, r, 1] for r in resources) + demand_data[prod][1])
        @constraint(model, demand_data[prod][1] == -s[prod, 1] + sum(x[prod, r, 1] for r in resources) + b[prod, 1])
    end

    for prod in products, t in 2:T
        @constraint(model, s[prod, t] >= (s[prod, t-1] + sum(x[prod, r, t] for r in resources)) - (f[prod, t-1] + demand_data[prod][t] + b[prod, t-1]))
        @constraint(model, b[prod, t] >= -(s[prod, t-1] + sum(x[prod, r, t] for r in resources)) + (f[prod, t-1] + demand_data[prod][t] + b[prod, t-1]))
        @constraint(model, demand_data[prod][t] == s[prod, t-1] + sum(x[prod, r, t] for r in resources) - f[prod, t-1] - s[prod, t] - b[prod, t-1] + b[prod, t])
    end

    for prod in products, t in 1:T
        @constraint(model, f[prod, t] == s[prod, t] * spoilage_rates[prod])
    end

    for t in 1:T
        @constraint(model, sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r, t] for prod in products, r in resources) + sum(inventory_carbon[prod] * s[prod, t] for prod in products) <= caps[t])
    end

    for r in resources, t in 1:T
        @constraint(model, sum(x[prod, r, t] for prod in products) <= production_caps[r])
    end

    @expression(model, total_cost, sum(product_resource_params[prod][r]["prod_cost"] * x[prod, r, t] for prod in products, r in resources, t in 1:T) + sum(inventory_costs[prod] * s[prod, t] for prod in products, t in 1:T) + sum(waste_costs[prod] * f[prod, t] for prod in products, t in 1:T) + sum(backorder_costs[prod] * b[prod, t] for prod in products, t in 1:T))

    @expression(model, total_carbon, sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r, t] for prod in products, r in resources, t in 1:T) + sum(inventory_carbon[prod] * s[prod, t] for prod in products, t in 1:T))

    @objective(model, Min, total_cost)
    optimize!(model)

    return JuMP.value(total_cost), JuMP.value(total_carbon), JuMP.value.(b), JuMP.termination_status(model)
end

# test different caps
caps_range = 8700:-5:7800
results = []

for cap in caps_range
    caps = [cap for t in 1:T]
    total_cost, total_carbon, backorder, status = deterministic_product_production(caps)
    push!(results, (cap = cap, total_cost = total_cost, total_carbon = total_carbon, backorder = sum(backorder), optimality = status))
end

# Save results
results_df = DataFrame(results)
CSV.write("model1_carbon_cap_results.csv", results_df)

# Print results
println("Carbon Cap, Total Cost, Total Carbon, Total Backorder, Optimality")
for result in results
    println("$(result.cap), $(result.total_cost), $(result.total_carbon), $(result.backorder), $(result.optimality)")
end

# Plotting
results_df = CSV.read("model1_carbon_cap_results.csv", DataFrame)
carbon_cap = results_df.cap
total_cost = results_df.total_cost
total_carbon = results_df.total_carbon

plot1 = plot(carbon_cap, total_cost, xlabel = "Carbon Cap", ylabel = "Total Cost", label = "Total Cost", linewidth = 2, color = :blue)
savefig(plot1, joinpath(@__DIR__, "model1_Total_Cost_vs_Carbon_Cap.png"))
display(plot1)

plot2 = plot(carbon_cap, total_carbon, xlabel = "Carbon Cap", ylabel = "Total Carbon", label = "Total Carbon", linewidth = 2, color = :red)
savefig(plot2, joinpath(@__DIR__, "model1_Total_Carbon_vs_Carbon_Cap.png"))
display(plot2)

plot3 = plot(carbon_cap, [total_cost total_carbon], xlabel = "Carbon Cap", ylabel = "Value", label = ["Total Cost" "Total Carbon"], linewidth = 2, color = [:blue :red])
savefig(plot3, joinpath(@__DIR__, "model1_Combined_vs_Carbon_Cap.png"))
display(plot3)

function is_pareto_optimal(i, costs, carbons)
    for j in 1:length(costs)
        if costs[j] < costs[i] && carbons[j] <= carbons[i]
            return false
        elseif carbons[j] < carbons[i] && costs[j] <= costs[i]
            return false
        end
    end
    return true
end

n = length(total_cost)
pareto_indices = [i for i in 1:n if is_pareto_optimal(i, total_cost, total_carbon)]
pareto_carbons = total_carbon[pareto_indices]
pareto_costs = total_cost[pareto_indices]
order = sortperm(pareto_carbons)
sorted_pareto_carbons = pareto_carbons[order]
sorted_pareto_costs = pareto_costs[order]

pPareto = scatter(total_carbon, total_cost, xlabel = "Total Carbon", ylabel = "Total Cost", label = "All Solutions", color = :gray, alpha = 0.6, markersize = 3)
scatter!(pPareto, sorted_pareto_carbons, sorted_pareto_costs, label = "Pareto Frontier", color = :red, markersize = 3)
savefig(pPareto, joinpath(@__DIR__, "model1_Pareto_Frontier.png"))
display(pPareto)

println("Pareto Optimal Points (by index):")
for idx in pareto_indices
    println("Index: $idx, Carbon Cap: $(carbon_cap[idx]), Total Cost: $(total_cost[idx]), Total Carbon: $(total_carbon[idx])")
end

results_df[!, :pareto_optimal] = falses(n)
results_df.pareto_optimal[pareto_indices] .= true
println("\nSummary of results with Pareto optimality flag:")
display(results_df)
CSV.write("model1_carbon_cap_results_pareto_check.csv", results_df)



