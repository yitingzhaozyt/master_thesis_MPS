using SDDP, HiGHS, Random, Printf, Gurobi, Test, Statistics, Plots, LaTeXStrings


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

# 🌟 **优化模型（带随机需求）**
function biobjective_product_production()
    model = SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # 定义状态变量和决策变量
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)   # carryover
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)   # 缺货
        @variable(sp, x[products, resources] >= 0)  # 生产量
        @variable(sp, f[products] >= 0)  # 废弃量
        @variable(sp, s[products] >= 0)  # 库存
        @variable(sp, ω_demand[products])  # 随机需求

        # 初始化双目标子问题
        SDDP.initialize_biobjective_subproblem(sp)

        # 随机需求扰动（保留原逻辑）
        Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for product in products
                fix(ω_demand[product], demand_data[product][t] * (1 + ω))
            end

            # 🌟 定义双目标函数
            cost = sum(product_resource_params[product][r]["prod_cost"] * x[product, r] 
                        for product in products, r in resources) +
                   sum(inventory_costs[product] * s[product] for product in products) +
                   sum(waste_costs[product] * f[product] for product in products) +
                   sum(backorder_costs[product] * b[product].out for product in products)

            carbon = sum(product_resource_params[product][r]["carbon_emission"] * x[product, r] 
                         for product in products, r in resources) +
                     sum(inventory_carbon[product] * s[product] for product in products)

            SDDP.set_biobjective_functions(sp, cost, carbon)
        end

        # 供需平衡约束
        for product in products
            @constraint(sp, s[product] >= sum(x[product, r] for r in resources) - ω_demand[product] + q[product].in - b[product].in)
            @constraint(sp, b[product].out >= -sum(x[product, r] for r in resources) + ω_demand[product] - q[product].in + b[product].in)
            @constraint(sp, f[product] == s[product] * spoilage_rates[product])
            @constraint(sp, q[product].out == s[product] - f[product])
            @constraint(sp, ω_demand[product] == -b[product].in + q[product].in + sum(x[product, r] for r in resources) - s[product] + b[product].out)
        end

        # 新增：资源产能约束。对于每个资源，每个阶段生产总量不能超过该资源的产能
        for r in resources
            @constraint(sp, sum(x[product, r] for product in products) <= resource_capacity[r])
        end
    end

    # 训练双目标模型
    pareto_weights = SDDP.train_biobjective(
        model;
        solution_limit = 10,
        iteration_limit = 10,
    )
    solutions = [(k, v) for (k, v) in pareto_weights]
    sort!(solutions; by = x -> x[1])
    @test length(solutions) == 10

    # 梯度测试（保留原测试逻辑）
    #gradient(a, b) = (b[2] - a[2]) / (b[1] - a[1])
   # grad = Inf
   # for i in 1:(length(solutions)-1)
   #     new_grad = gradient(solutions[i], solutions[i+1])
  #      @test new_grad < grad
   #     grad = new_grad
  #  end

    println("Lower bound: ", SDDP.calculate_bound(model))

    return model, solutions
end

# Run the biobjective model.
model, solutions_summary = biobjective_product_production()

# Extract the lambda weights and corresponding cost-to-go values
lambdas = [sol[1] for sol in solutions_summary]
V1K = [sol[2] for sol in solutions_summary]

# Plot the cost-to-go as a function of lambda
p1 = Plots.plot(lambdas, V1K,
    marker = (:circle, 4),
    xlabel = L"Weight $\lambda$",
    ylabel = L"Cost-to-go V_{1}^{K}(\lambda)",
    legend = false,
    xformatter = :plain,
    yformatter = :plain,)
display(p1)
Plots.savefig(p1, "model4_weights_cost_to_go_high.png")

# Run one simulation and print the keys of the first stage dictionary.
sim_debug = SDDP.simulate(model, 1)
println("Available keys in first stage dictionary: ", keys(sim_debug[1][1]))

# ============================================================================ 
# 🔄 **Simulation under fixed trade-off weights**
#
# Now we simulate the trained model under three different lambda values: 0.3125, 0.2, and 0.5.
# For each lambda, we set the weight using SDDP.set_trade_off_weight(model, weight)
# and then simulate 100 scenarios. The simulation costs are plotted for each lambda.
# ============================================================================

# Define the lambda values and number of simulation scenarios.
lambda_values = [0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
num_simulations = 1000


# ===== Helper Function to Compute Global Cost and Carbon =====
function compute_total_cost_and_carbon(sim, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, 
    waste_costs, backorder_costs)
    
    total_cost = 0.0
    total_carbon = 0.0
    # Loop over each stage in the simulation trajectory
    for t in 1:T
        # Production: iterate over all products and resources
        for product in products, r in resources
            prod_val = sim[t][:x][product, r]
            total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
            total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
        end
        # Inventory, waste, and backorder components:
        for product in products
            total_cost += inventory_costs[product] * sim[t][:s][product]
            total_carbon += inventory_carbon[product] * sim[t][:s][product]
            total_cost += waste_costs[product] * sim[t][:f][product]
            total_cost += backorder_costs[product] * sim[t][:b][product].out
        end
    end
    return total_cost, total_carbon
end

# Prepare scatter plots:
plot_cost = Plots.plot(
    xlabel = "Global Total Cost", ylabel = "Weighted Objective Value", legend = :bottomleft, xformatter = :plain,
    yformatter = :plain,)
plot_carbon = Plots.plot(
    xlabel = "Global Total Carbon", ylabel = "Weighted Objective Value", legend = :bottomright, xformatter = :plain,
    yformatter = :plain,)

# Loop over each λ value:
for λ in lambda_values
    # Set the trade-off weight in your biobjective model.
    SDDP.set_trade_off_weight(model, λ)
    
    # Run simulations while saving decision variables.
    sim_results = SDDP.simulate(model, num_simulations, [:q, :b, :x, :f, :s, :ω_demand])
    
    # Prepare arrays to hold computed metrics.
    cost_vals = Float64[]
    carbon_vals = Float64[]
    weighted_objs = Float64[]
    
    # Process each simulation run.
    for sim_run in sim_results
        # Compute global total cost and carbon using your helper function.
        tot_cost, tot_carbon = compute_total_cost_and_carbon(sim_run, products, resources, T, 
            product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
        push!(cost_vals, tot_cost)
        push!(carbon_vals, tot_carbon)
        # The weighted objective is the cumulative sum of stage objectives.
        push!(weighted_objs, sum(stage[:stage_objective] for stage in sim_run))
    end
    
    # Add scatter points for this λ value.
    Plots.scatter!(plot_cost, cost_vals, weighted_objs,
        marker = (:circle, 3), label = "λ = $(λ)")
    Plots.scatter!(plot_carbon, carbon_vals, weighted_objs,
        marker = (:circle, 3), label = "λ = $(λ)")
end

# Display and save the plots.
display(plot_cost)
Plots.savefig(plot_cost, "model4_biobjective_simulation_weighted_vs_global_total_cost_high.png")

display(plot_carbon)
Plots.savefig(plot_carbon, "model4_biobjective_simulation_weighted_vs_global_total_carbon_high.png")







# 创建散点图：X轴是 carbon，Y轴是 cost
carbon_vs_cost_plot = Plots.plot(
    xlabel = "Total Carbon Emissions",
    ylabel = "Total Cost",
    #title = "Cost vs Carbon under Different λ",
    legend = :topright,
    xformatter = :plain,
    yformatter = :plain,
)

# 用不同颜色画出每个 λ 下的散点
for λ in lambda_values
    SDDP.set_trade_off_weight(model, λ)
    sim_results = SDDP.simulate(model, num_simulations, [:q, :b, :x, :f, :s, :ω_demand])

    carbon_vals = Float64[]
    cost_vals = Float64[]

    for sim_run in sim_results
        tot_cost, tot_carbon = compute_total_cost_and_carbon(sim_run, products, resources, T,
            product_resource_params, inventory_costs, inventory_carbon,
            waste_costs, backorder_costs)
        push!(cost_vals, tot_cost)
        push!(carbon_vals, tot_carbon)
    end

    # 在同一图上叠加散点
    scatter!(carbon_vs_cost_plot, carbon_vals, cost_vals,
        marker = (:circle, 3), label = "λ = $(λ)")
end

display(carbon_vs_cost_plot)
savefig(carbon_vs_cost_plot, "model4_carbon_vs_cost_by_lambda_high.png")





println("\\begin{table}[H]")
println("\\centering")
println("\\begin{tabular}{cccc}")
println("\\hline")
println("\\textbf{Weight} & \\textbf{Avg. Cost} & \\textbf{Avg. Carbon} & \\textbf{Weighted Objective} \\\\")
println("\\hline")

for λ in lambda_values
    SDDP.set_trade_off_weight(model, λ)
    sim_results = SDDP.simulate(model, num_simulations, [:q, :b, :x, :f, :s, :ω_demand])

    cost_vals = Float64[]
    carbon_vals = Float64[]
    weighted_vals = Float64[]

    for sim_run in sim_results
        tot_cost, tot_carbon = compute_total_cost_and_carbon(sim_run, products, resources, T,
            product_resource_params, inventory_costs, inventory_carbon,
            waste_costs, backorder_costs)

        push!(cost_vals, tot_cost)
        push!(carbon_vals, tot_carbon)
        push!(weighted_vals, sum(stage[:stage_objective] for stage in sim_run))
    end

    avg_cost = mean(cost_vals)
    avg_carbon = mean(carbon_vals)
    avg_weighted = mean(weighted_vals)

    @printf("%.4f  & %.2f  & %.2f  & %.2f \\\\\n", λ, avg_cost, avg_carbon, avg_weighted)
end

println("\\hline")
println("\\end{tabular}")
println("\\caption{Weight, Average Cost, Average Carbon Emissions, and Weighted Objective Values}")
println("\\label{tab:weight_cost_carbon}")
println("\\end{table}")




