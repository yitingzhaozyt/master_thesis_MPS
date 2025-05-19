using JuMP, Gurobi, Random, Plots

T = 12
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

# Set random seed
Random.seed!(42)

# Generate random demand
demand_data = Dict{String, Vector{Int}}()
for prod in products
    demand_data[prod] = [rand(100:250) for t in 1:T]
end

# Define production cost and carbon emission ranges
ranges = Dict(
    "A" => Dict("prod_cost" => (4.0, 8.0), "carbon_emission" => (10.0, 12.0)),
    "B" => Dict("prod_cost" => (8.0, 10.0), "carbon_emission" => (8.0, 10.0)),
    "C" => Dict("prod_cost" => (10.0, 14.0), "carbon_emission" => (7.0, 8.0))
)

# Generate product-resource parameters
product_resource_params = Dict()
for product in products
    product_resource_params[product] = Dict()
    for resource in resources
        cost_range = ranges[resource]["prod_cost"]
        co2_range = ranges[resource]["carbon_emission"]
        prod_cost = round(rand() * (cost_range[2] - cost_range[1]) + cost_range[1]; digits=2)
        carbon_emission = round(rand() * (co2_range[2] - co2_range[1]) + co2_range[1]; digits=3)
        product_resource_params[product][resource] = Dict(
            "prod_cost" => prod_cost,
            "carbon_emission" => carbon_emission
        )
    end
end

# Generate inventory cost, spoilage, etc.
inventory_cost_range = (1.0, 3.0)
inventory_carbon_range = (0.5, 2.0)
backorder_cost_range = (8.0, 15.0)
waste_cost_range = (0.1, 1.0)
spoilage_rate_range = (0.05, 0.14)

inventory_costs = Dict(p => round(rand() * (inventory_cost_range[2] - inventory_cost_range[1]) + inventory_cost_range[1]; digits=2) for p in products)
inventory_carbon = Dict(p => round(rand() * (inventory_carbon_range[2] - inventory_carbon_range[1]) + inventory_carbon_range[1]; digits=3) for p in products)
backorder_costs = Dict(p => round(rand() * (backorder_cost_range[2] - backorder_cost_range[1]) + backorder_cost_range[1]; digits=2) for p in products)
waste_costs = Dict(p => round(rand() * (waste_cost_range[2] - waste_cost_range[1]) + waste_cost_range[1]; digits=2) for p in products)
spoilage_rates = Dict(p => round(rand() * (spoilage_rate_range[2] - spoilage_rate_range[1]) + spoilage_rate_range[1]; digits=3) for p in products)

# ===============================
# Pretty Print Parameters
# ===============================

function print_product_parameters(product_resource_params, inventory_costs, inventory_carbon, backorder_costs, waste_costs, spoilage_rates)
    println("# Updated production parameters (mapping order as follows):")
    println("product_resource_params = Dict(")
    ordered_products = ["product1", "product2", "product3", "product4", "product5"]
    
    for (i, prod) in enumerate(ordered_products)
        println("    \"$prod\" => Dict(")
        res_keys = ["A", "B", "C"]
        for (j, r) in enumerate(res_keys)
            entry = "        \"$r\" => Dict(\"prod_cost\" => $(product_resource_params[prod][r]["prod_cost"]), \"carbon_emission\" => $(product_resource_params[prod][r]["carbon_emission"]))"
            if j < length(res_keys)
                println(entry, ",")
            else
                println(entry)  # 最后一个 resource 不加逗号
            end
        end
        if i < length(ordered_products)
            println("    ),")
        else
            println("    )")  # 最后一个 product 不加逗号
        end
    end
    println(")\n")
    
    # 🔥 这里改成一横排
    function print_flat_dict(name, dict)
        print("$name = Dict(")
        for (i, prod) in enumerate(ordered_products)
            print("\"$prod\" => $(dict[prod])")
            if i < length(ordered_products)
                print(", ")
            end
        end
        println(")\n")
    end
    
    print_flat_dict("inventory_costs", inventory_costs)
    print_flat_dict("inventory_carbon", inventory_carbon)
    print_flat_dict("backorder_costs", backorder_costs)
    print_flat_dict("waste_costs", waste_costs)
    print_flat_dict("spoilage_rates", spoilage_rates)
end



# Print the generated parameters
print_product_parameters(product_resource_params, inventory_costs, inventory_carbon, backorder_costs, waste_costs, spoilage_rates)



####################################################################
####################################################################
####################################################################


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