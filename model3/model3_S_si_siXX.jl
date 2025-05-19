using SDDP, HiGHS, Random, Printf, Gurobi, Plots, Statistics

# ---------------------------
# Data Definitions
# ---------------------------
# Update product names: use generic names instead of the original prod names.
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]
T = 12  # 时间周期

# Generate random demand data for each product and period.
Random.seed!(42)
demand_data = Dict{String, Vector{Int}}()
for prod in products
    demand_data[prod] = [rand(100:250) for t in 1:T]
end

# Carbon cap per period (example: each period capped at 12000 units).
caps = [12000 for t in 1:T]

# Updated production parameters – keys now use generic product names.
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

cd(@__DIR__)

inventory_costs = Dict("product1" => 1.6, "product2" => 2.06, "product3" => 2.67, "product4" => 1.93, "product5" => 1.23)
inventory_carbon = Dict("product1" => 1.172, "product2" => 0.507, "product3" => 1.69, "product4" => 1.924, "product5" => 0.904)
backorder_costs = Dict("product1" => 9.79, "product2" => 14.15, "product3" => 10.23, "product4" => 12.37, "product5" => 12.5)
waste_costs = Dict("product1" => 0.53, "product2" => 0.27, "product3" => 0.82, "product4" => 0.32, "product5" => 0.14)
spoilage_rates = Dict("product1" => 0.099, "product2" => 0.126, "product3" => 0.131, "product4" => 0.112, "product5" => 0.111)


# Define production capacity limits per resource (applied per period)
production_caps = Dict("A" => 350, "B" => 350, "C" => 350)

# ---------------------------
# Build SDDP Model
# ---------------------------
function stochastic_food_production()
    model = SDDP.LinearPolicyGraph(
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # State variables
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)

        # Decision variables
        @variable(sp, x[products, resources] >= 0)
        @variable(sp, f[products] >= 0)
        @variable(sp, s[products] >= 0)
        @variable(sp, ω_demand[products])

        # Demand uncertainty: add random shocks
        Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        #Ω = [0,0,0,0,0]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for prod in products
                fix(ω_demand[prod], demand_data[prod][t] * (1 + ω))
            end
        end

        for prod in products
            @constraint(sp, s[prod] >= sum(x[prod, r] for r in resources) - ω_demand[prod] + q[prod].in - b[prod].in)
            @constraint(sp, b[prod].out >= -sum(x[prod, r] for r in resources) + ω_demand[prod] - q[prod].in + b[prod].in)
            @constraint(sp, f[prod] == s[prod] * spoilage_rates[prod])
            @constraint(sp, q[prod].out == s[prod] - f[prod])
            @constraint(sp, ω_demand[prod] == -b[prod].in + q[prod].in + sum(x[prod, r] for r in resources) - s[prod] + b[prod].out)
        end

        # Capacity constraint
        for r in resources
            @constraint(sp, sum(x[prod, r] for prod in products) <= production_caps[r])
        end

        # Carbon constraint
        @constraint(sp,
            sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r] for prod in products, r in resources) +
            sum(inventory_carbon[prod] * s[prod] for prod in products) <= caps[t]
        )

        # Objective function: minimize all costs
        @stageobjective(sp,
            sum(product_resource_params[prod][r]["prod_cost"] * x[prod, r] for prod in products, r in resources) +
            sum(inventory_costs[prod] * s[prod] for prod in products) +
            sum(waste_costs[prod] * f[prod] for prod in products) +
            sum(backorder_costs[prod] * b[prod].out for prod in products)
        )
    end

    SDDP.train(model; iteration_limit = 10)
    println("Lower bound: ", SDDP.calculate_bound(model))
    return model
end

model = stochastic_food_production()

# ---------------------------
# Simulation and Visualization
# ---------------------------
simulations = SDDP.simulate(model, 100, [:q, :b, :x, :f, :s, :ω_demand])

# Static Plot: Avg production vs demand
total_production = zeros(length(simulations), T)
for i in 1:length(simulations)
    for t in 1:T
        total_production[i, t] = sum(simulations[i][t][:x])
    end
end
avg_production = [mean(total_production[:, t]) for t in 1:T]
demand_sim = [sum(simulations[1][t][:ω_demand]) for t in 1:T]

p_static = plot(1:T, demand_sim, label="Demand", xlabel="Period", ylabel="Total", lw=2)
plot!(1:T, avg_production, label="Avg Production", lw=2)

# Interactive Spaghetti Plot
spag_plot = SDDP.SpaghettiPlot(simulations)

SDDP.add_spaghetti(spag_plot; title="Total Production") do sim
    sum(sim[:x][prod, r] for prod in products, r in resources)
end
SDDP.add_spaghetti(spag_plot; title="Total Demand") do sim
    sum(sim[:ω_demand][prod] for prod in products)
end
SDDP.add_spaghetti(spag_plot; title="Total Inventory") do sim
    sum(sim[:s][prod] for prod in products)
end
SDDP.plot(spag_plot, "spaghetti_food_plot.html"; open=false)

# ---------------------------
# Cost & Emission Calculation
# ---------------------------
function compute_total_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, 
    waste_costs, backorder_costs)
    
    total_cost = 0.0
    total_carbon = 0.0
    sim = simulations[42]  # One simulation path

    for t in 1:T
        for prod in products, r in resources
            xval = sim[t][:x][prod, r]
            total_cost += product_resource_params[prod][r]["prod_cost"] * xval
            total_carbon += product_resource_params[prod][r]["carbon_emission"] * xval
        end
        for prod in products
            total_cost += inventory_costs[prod] * sim[t][:s][prod]
            total_carbon += inventory_carbon[prod] * sim[t][:s][prod]
            total_cost += waste_costs[prod] * sim[t][:f][prod]
            total_cost += backorder_costs[prod] * sim[t][:b][prod].out
        end
    end

    return total_cost, total_carbon
end

tc, tcarbon = compute_total_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)

@printf("Under the optimal solution, the total cost is: %.2f\n", tc)
@printf("Under the optimal solution, the total carbon emissions are: %.2f\n", tcarbon)
