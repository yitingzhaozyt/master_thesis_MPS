using JuMP, Gurobi, Random, DataFrames, CSV, Plots

# data
T = 12

Random.seed!(42)
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

demand_data = Dict(prod => [rand(100:250) for _ in 1:T] for prod in products)

production_caps = Dict("A" => 350, "B" => 350, "C" => 350)

product_resource_params = Dict(
    "product1" => Dict("A" => Dict("prod_cost" => 6.62, "carbon_emission" => 10.562),
                       "B" => Dict("prod_cost" => 8.73, "carbon_emission" => 9.695),
                       "C" => Dict("prod_cost" => 12.74, "carbon_emission" => 7.699)),
    "product2" => Dict("A" => Dict("prod_cost" => 5.41, "carbon_emission" => 10.132),
                       "B" => Dict("prod_cost" => 8.39, "carbon_emission" => 9.928),
                       "C" => Dict("prod_cost" => 11.44, "carbon_emission" => 7.779)),
    "product3" => Dict("A" => Dict("prod_cost" => 6.85, "carbon_emission" => 11.583),
                       "B" => Dict("prod_cost" => 8.15, "carbon_emission" => 9.392),
                       "C" => Dict("prod_cost" => 10.4, "carbon_emission" => 7.623)),
    "product4" => Dict("A" => Dict("prod_cost" => 7.79, "carbon_emission" => 10.871),
                       "B" => Dict("prod_cost" => 9.95, "carbon_emission" => 9.875),
                       "C" => Dict("prod_cost" => 12.96, "carbon_emission" => 7.416)),
    "product5" => Dict("A" => Dict("prod_cost" => 7.08, "carbon_emission" => 11.175),
                       "B" => Dict("prod_cost" => 8.7, "carbon_emission" => 8.173),
                       "C" => Dict("prod_cost" => 12.81, "carbon_emission" => 7.558))
)

inventory_costs = Dict("product1" => 1.6, "product2" => 2.06, "product3" => 2.67, "product4" => 1.93, "product5" => 1.23)
inventory_carbon = Dict("product1" => 1.172, "product2" => 0.507, "product3" => 1.69, "product4" => 1.924, "product5" => 0.904)
backorder_costs = Dict("product1" => 9.79, "product2" => 14.15, "product3" => 10.23, "product4" => 12.37, "product5" => 12.5)
waste_costs = Dict("product1" => 0.53, "product2" => 0.27, "product3" => 0.82, "product4" => 0.32, "product5" => 0.14)
spoilage_rates = Dict("product1" => 0.099, "product2" => 0.126, "product3" => 0.131, "product4" => 0.112, "product5" => 0.111)

cd(@__DIR__)

# model
function deterministic_product_production(lambda)
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

    for r in resources, t in 1:T
        @constraint(model, sum(x[prod, r, t] for prod in products) <= production_caps[r])
    end

    @expression(model, total_cost,
        sum(product_resource_params[prod][r]["prod_cost"] * x[prod, r, t] for prod in products, r in resources, t in 1:T) +
        sum(inventory_costs[prod] * s[prod, t] for prod in products, t in 1:T) +
        sum(waste_costs[prod] * f[prod, t] for prod in products, t in 1:T) +
        sum(backorder_costs[prod] * b[prod, t] for prod in products, t in 1:T)
    )

    @expression(model, total_carbon,
        sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r, t] for prod in products, r in resources, t in 1:T) +
        sum(inventory_carbon[prod] * s[prod, t] for prod in products, t in 1:T)
    )

    @objective(model, Min, lambda * total_cost + (1 - lambda) * total_carbon)

    optimize!(model)

    return JuMP.value(total_cost), JuMP.value(total_carbon), JuMP.objective_value(model)
end

# run different lambda
lambda_values = 0:0.1:1
results = DataFrame(lambda=Float64[], total_cost=Float64[], total_carbon=Float64[], objective_value=Float64[])

for lambda in lambda_values
    total_cost, total_carbon, objective_value = deterministic_product_production(lambda)
    push!(results, (lambda, total_cost, total_carbon, objective_value))
end

# 保存
CSV.write("model2_lambda_tradeoff_results.csv", results)

using Printf 

# Trade-off curves
tradeoff_plot = plot(
    results.lambda, results.total_cost,
    label="Total Cost", xlabel="Lambda", ylabel="Value",
    linewidth=2,
    xformatter = x -> @sprintf("%.1f", x),
    yformatter = y -> string(round(Int, y))
)
plot!(
    tradeoff_plot, results.lambda, results.total_carbon,
    label="Total Carbon", linewidth=2
)

display(tradeoff_plot)
savefig(tradeoff_plot, "model2_lambda_tradeoff_plot_annotated.png")

# Pareto front 
pareto_plot = plot(
    results.total_carbon, results.total_cost,
    xlabel="Total Carbon", ylabel="Total Cost",
    label="Pareto Frontier", linewidth=2, marker=:circle,
    markersize=3,
    xformatter = x -> string(round(Int, x)),
    yformatter = y -> string(round(Int, y))
)
savefig(pareto_plot, "model2_lambda_pareto_frontier.png")
display(pareto_plot)




lambda_objective_plot = plot(
    results.lambda, results.objective_value,
    xlabel = "Lambda",
    ylabel = "Objective Value",
    #title = "Objective Value vs Lambda",
    label = "Objective Value",
    linewidth = 2,
    xformatter = x -> @sprintf("%.1f", x),
    yformatter = y -> string(round(Int, y))
)


savefig(lambda_objective_plot, "model2_lambda_vs_objective_value.png")
display(lambda_objective_plot)
