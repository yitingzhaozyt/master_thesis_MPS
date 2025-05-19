using JuMP, Gurobi, Random, Plots

# ===============================
# Data definition
# ===============================

T = 12
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

# Resource capacity
resource_cap = Dict("A" => 350, "B" => 350, "C" => 350)

# Carbon cap per period
carbon_cap = [8000 for t in 1:T]  

# Set random seed
Random.seed!(42)

# Generate random demand
demand_data = Dict{String, Vector{Int}}()
for prod in products
    demand_data[prod] = [rand(100:250) for t in 1:T]
end

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


# ===============================
# Main Deterministic Production Model
# ===============================

function deterministic_product_production()
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)

    @variable(model, s[products, 1:T] >= 0)       # inventory
    @variable(model, b[products, 1:T] >= 0)       # backorder
    @variable(model, x[products, resources, 1:T] >= 0)  # production
    @variable(model, f[products, 1:T] >= 0)       # waste

    # Period 1 balance
    for prod in products
        @constraint(model, demand_data[prod][1] == -s[prod, 1] + sum(x[prod, r, 1] for r in resources) + b[prod, 1])
    end

    # Period 2 to T
    for prod in products, t in 2:T
        @constraint(model, demand_data[prod][t] == s[prod, t-1] + sum(x[prod, r, t] for r in resources) - f[prod, t-1] - s[prod, t] - b[prod, t-1] + b[prod, t])
    end

    # Capacity limit for each resource
    for r in resources, t in 1:T
        @constraint(model, sum(x[prod, r, t] for prod in products) <= resource_cap[r])
    end

    # Spoilage waste link
    for prod in products, t in 1:T
        @constraint(model, f[prod, t] == s[prod, t] * spoilage_rates[prod])
    end

    # Carbon emission limit per period
    for t in 1:T
        @constraint(model,
            sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r, t] for prod in products, r in resources) +
            sum(inventory_carbon[prod] * s[prod, t] for prod in products) <= carbon_cap[t]
        )
    end

    # Total cost and total carbon expressions
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

    @objective(model, Min, total_cost)
    optimize!(model)

    println("\n=== Resource Utilization Summary ===")
    header = rpad("Period", 8) * 
            rpad("A Prod", 12) * rpad("A Util(%)", 12) *
            rpad("B Prod", 12) * rpad("B Util(%)", 12) *
            rpad("C Prod", 12) * rpad("C Util(%)", 12)
    println(header)
    println("-"^length(header))

    for t in 1:T
        prod_A = sum(JuMP.value(x[prod, "A", t]) for prod in products)
        prod_B = sum(JuMP.value(x[prod, "B", t]) for prod in products)
        prod_C = sum(JuMP.value(x[prod, "C", t]) for prod in products)

        util_A = prod_A / resource_cap["A"] * 100
        util_B = prod_B / resource_cap["B"] * 100
        util_C = prod_C / resource_cap["C"] * 100

        println(rpad("$(t)", 8),
                rpad(string(round(prod_A, digits=2)), 12), rpad(string(round(util_A, digits=2)), 12),
                rpad(string(round(prod_B, digits=2)), 12), rpad(string(round(util_B, digits=2)), 12),
                rpad(string(round(prod_C, digits=2)), 12), rpad(string(round(util_C, digits=2)), 12))
    end






    println("Termination Status: ", JuMP.termination_status(model))
    println("Objective Value (Total Cost): ", JuMP.objective_value(model))
    println("Total Carbon (for reference): ", JuMP.value(total_carbon))

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

    println("\nProduction quantities from A, B, C and Backorder for each product and period:")
    for t in 1:T
        carbon_cap_t = carbon_cap[t]
        println("Period $t, Carbon Cap $carbon_cap_t:")
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

    println("\nCapacity usage for each resource in each period:")
    for t in 1:T
        println("Period $t:")
        for r in resources
            resource_usage = sum(JuMP.value(x[prod, r, t]) for prod in products)
            println("  Resource $r: Usage = $resource_usage, Capacity = $(resource_cap[r])")
        end
    end

    # === Plot Demand Data ===
    pDemand = plot(xlabel="Period", ylabel="Demand", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pDemand, 1:T, demand_data[prod], label=prod, marker=:circle)
    end
    savefig(pDemand, joinpath(@__DIR__, "demand_plot.png"))
    display(pDemand)

    # === Extract Production Data ===
    prod_A = Dict{String, Vector{Float64}}()
    prod_B = Dict{String, Vector{Float64}}()
    prod_C = Dict{String, Vector{Float64}}()

    for prod in products
        prod_A[prod] = [JuMP.value(x[prod, "A", t]) for t in 1:T]
        prod_B[prod] = [JuMP.value(x[prod, "B", t]) for t in 1:T]
        prod_C[prod] = [JuMP.value(x[prod, "C", t]) for t in 1:T]
    end

    # === Plot Production Data for Resource A ===
    pA = plot(xlabel="Period", ylabel="Production Quantity (A)", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pA, 1:T, prod_A[prod], label=prod, marker=:circle)
    end
    savefig(pA, joinpath(@__DIR__, "production_resource_A.png"))
    display(pA)

    # === Plot Production Data for Resource B ===
    pB = plot(xlabel="Period", ylabel="Production Quantity (B)", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pB, 1:T, prod_B[prod], label=prod, marker=:circle)
    end
    savefig(pB, joinpath(@__DIR__, "production_resource_B.png"))
    display(pB)

    # === Plot Production Data for Resource C ===
    pC = plot(xlabel="Period", ylabel="Production Quantity (C)", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pC, 1:T, prod_C[prod], label=prod, marker=:circle)
    end
    savefig(pC, joinpath(@__DIR__, "production_resource_C.png"))
    display(pC)

    # === Extract Inventory, Waste, and Backorder Data ===
    inventory_data = Dict{String, Vector{Float64}}()
    waste_data = Dict{String, Vector{Float64}}()
    backorder_data = Dict{String, Vector{Float64}}()

    for prod in products
        inventory_data[prod] = [JuMP.value(s[prod, t]) for t in 1:T]
        waste_data[prod] = [JuMP.value(f[prod, t]) for t in 1:T]
        backorder_data[prod] = [JuMP.value(b[prod, t]) for t in 1:T]
    end

    # === Plot Inventory Data ===
    pInv = plot(xlabel="Period", ylabel="Inventory", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pInv, 1:T, inventory_data[prod], label=prod, marker=:circle)
    end
    savefig(pInv, joinpath(@__DIR__, "inventory_plot.png"))
    display(pInv)

    # === Plot Waste Data ===
    pWaste = plot(xlabel="Period", ylabel="Waste Volume", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pWaste, 1:T, waste_data[prod], label=prod, marker=:circle)
    end
    savefig(pWaste, joinpath(@__DIR__, "waste_plot.png"))
    display(pWaste)

    # === Plot Backorder Data ===
    pBackorder = plot(xlabel="Period", ylabel="Backorder", legend=:outerright, xticks=1:T)
    for prod in products
        plot!(pBackorder, 1:T, backorder_data[prod], label=prod, marker=:circle)
    end
    savefig(pBackorder, joinpath(@__DIR__, "backorder_plot.png"))
    display(pBackorder)

    # === Plot Carbon Emissions Per Period ===
    carbon_emissions_per_period = []
    for t in 1:T
        carbon_t = sum(product_resource_params[prod][r]["carbon_emission"] * JuMP.value(x[prod, r, t]) for prod in products, r in resources) +
                sum(inventory_carbon[prod] * JuMP.value(s[prod, t]) for prod in products)
        push!(carbon_emissions_per_period, carbon_t)
    end

    pCarbon = plot(1:T, carbon_emissions_per_period,
        xlabel = "Period",
        ylabel = "Carbon Emissions",
        #title = "Carbon Emissions per Period",
        legend = false,
        marker = :circle,
        linewidth = 2,
        grid = true,
        xticks = 1:T,
        ylim = (0, maximum(carbon_emissions_per_period) * 1.1)
    )
    savefig(pCarbon, joinpath(@__DIR__, "carbon_emissions_per_period.png"))
    display(pCarbon)

    


    return JuMP.value(total_cost), JuMP.value(total_carbon), solution_summary(model)
end

# Run the model
total_cost, total_carbon, summary = deterministic_product_production()
