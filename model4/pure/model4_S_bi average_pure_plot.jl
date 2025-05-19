using SDDP, HiGHS, Random, Printf, Gurobi, Test, Statistics, Plots


T = 12
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

# Resource capacity
resource_capacity = Dict("A" => 350, "B" => 350, "C" => 350)


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



cd(@__DIR__)


# 🧩 主函数：建立并求解模型
function biobjective_product_production()
    model = SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, x[products, resources] >= 0)
        @variable(sp, f[products] >= 0)
        @variable(sp, s[products] >= 0)
        @variable(sp, ω_demand[products])

        SDDP.initialize_biobjective_subproblem(sp)

        
        Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for product in products
                fix(ω_demand[product], demand_data[product][t] * (1 + ω))
            end

            cost = sum(product_resource_params[product][r]["prod_cost"] * x[product, r] for product in products, r in resources) +
                   sum(inventory_costs[product] * s[product] for product in products) +
                   sum(waste_costs[product] * f[product] for product in products) +
                   sum(backorder_costs[product] * b[product].out for product in products)

            carbon = sum(product_resource_params[product][r]["carbon_emission"] * x[product, r] for product in products, r in resources) +
                     sum(inventory_carbon[product] * s[product] for product in products)

            SDDP.set_biobjective_functions(sp, cost, carbon)
        end

        for product in products
            @constraint(sp, s[product] >= sum(x[product, r] for r in resources) - ω_demand[product] + q[product].in - b[product].in)
            @constraint(sp, b[product].out >= -sum(x[product, r] for r in resources) + ω_demand[product] - q[product].in + b[product].in)
            @constraint(sp, f[product] == s[product] * spoilage_rates[product])
            @constraint(sp, q[product].out == s[product] - f[product])
            @constraint(sp, ω_demand[product] == -b[product].in + q[product].in + sum(x[product, r] for r in resources) - s[product] + b[product].out)
        end

        for r in resources
            @constraint(sp, sum(x[product, r] for product in products) <= resource_cap[r])
        end
    end

    pareto_weights = SDDP.train_biobjective(
        model;
        solution_limit = 10,
        iteration_limit = 10,
    )
    solutions = [(k, v) for (k, v) in pareto_weights]
    sort!(solutions; by = x -> x[1])

    # 检查 Pareto 曲线
    check_pareto_monotonicity(solutions)

    println("\nLower bound: ", SDDP.calculate_bound(model))

    # 画 Pareto 前沿
    x_vals = [s[1] for s in solutions]
    y_vals = [s[2] for s in solutions]

    p = plot(x_vals, y_vals, xlabel="Total Cost", ylabel="Total Carbon", title="Pareto Front", marker=:circle, legend=false)
    savefig(p, joinpath(@__DIR__, "pareto_front.png"))
    display(p)

    return model, solutions
end

# 调用主函数
model, solutions_summary = biobjective_product_production()
