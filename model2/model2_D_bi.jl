using JuMP, Gurobi, Random

# 基础数据
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

# 定义模型
function deterministic_product_production(caps)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)

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

    @objective(model, Min, 0.8 * total_cost + 0.2 * total_carbon)
    optimize!(model)

    println("Termination Status: ", JuMP.termination_status(model))
    println("Objective Value: ", JuMP.objective_value(model))
    println("Total Cost: ", JuMP.value(total_cost))
    println("Total Carbon: ", JuMP.value(total_carbon))

    println("\nDemand for each product and period:")
    for prod in products
        println("$prod: ", demand_data[prod])
    end

    println("\nInventory (s) for each product and period:")
    for prod in products
        println("$prod: ", JuMP.value.(s[prod, :]))
    end

    println("\nBackorder (b) for each product and period:")
    for prod in products
        println("$prod: ", JuMP.value.(b[prod, :]))
    end

    println("\nProduction quantities and related data for each product and period:")
    for t in 1:T
        println("Period $t:")
        for prod in products
            prod_a = JuMP.value(x[prod, "A", t])
            prod_b = JuMP.value(x[prod, "B", t])
            prod_c = JuMP.value(x[prod, "C", t])
            backorder = JuMP.value(b[prod, t])
            inventory = JuMP.value(s[prod, t])
            demand = demand_data[prod][t]
            waste_volume = JuMP.value(f[prod, t])
            println("  $prod: A ($prod_a), B ($prod_b), C ($prod_c), Inventory ($inventory), Waste ($waste_volume), Backorder ($backorder), Demand ($demand)")
        end
    end

    println("\nCarbon emissions per period:")
    for t in 1:T
        carbon_t = sum(product_resource_params[prod][r]["carbon_emission"] * JuMP.value(x[prod, r, t]) for prod in products, r in resources) +
                   sum(inventory_carbon[prod] * JuMP.value(s[prod, t]) for prod in products)
        println("Period $t: $carbon_t")
    end

    return JuMP.value.(s), JuMP.value.(b), JuMP.value.(x), JuMP.value(total_cost), JuMP.value(total_carbon), JuMP.objective_value(model)
end

# 运行
s, b, x, total_cost, total_carbon, objective_value = deterministic_product_production([0 for t in 1:T])
