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

# 🌟 **优化模型（带随机需求）**
function biobjective_product_production()
    model = SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # 定义状态变量和决策变量
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)   #carryover
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

            carbon = sum(product_resource_params[product][r]["carbon_emission"] * x[product, r] for product in products, r in resources) +
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

        # 新增：资源产能约束
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
    gradient(a, b) = (b[2] - a[2]) / (b[1] - a[1])
    grad = Inf
    for i in 1:(length(solutions)-1)
        new_grad = gradient(solutions[i], solutions[i+1])
        @test new_grad < grad
        grad = new_grad
    end

    println("Lower bound: ", SDDP.calculate_bound(model))
    return model, solutions
end

# Run the biobjective model.
model, solutions_summary = biobjective_product_production()






# ---------------------------
# Helper Functions for Simulation Evaluation
# ---------------------------
# Compute cost and carbon for a single simulation trajectory.
function compute_total_cost_and_carbon_single(sim, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
    
    total_cost = 0.0
    total_carbon = 0.0
    for t in 1:T
        for product in products, r in resources
            prod_val = sim[t][:x][product, r]
            total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
            total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
        end
        for product in products
            total_cost += inventory_costs[product] * sim[t][:s][product] +
                          waste_costs[product] * sim[t][:f][product] +
                          backorder_costs[product] * sim[t][:b][product].out
            total_carbon += inventory_carbon[product] * sim[t][:s][product]
        end
    end
    return total_cost, total_carbon
end

# Average the cost and carbon over all simulation replications.
function average_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
    
    costs = Float64[]
    carbons = Float64[]
    for sim in simulations
        tc, tcarbon = compute_total_cost_and_carbon_single(sim, products, resources, T, 
            product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
        push!(costs, tc)
        push!(carbons, tcarbon)
    end
    return mean(costs), mean(carbons)
end

# ---------------------------
# Compute Pareto Points Averaged over Simulations in 3D
# ---------------------------
function compute_pareto_points_3d(model, fixed_weights, products, resources, T,
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
    
    fixed_weight_points = Float64[]
    cost_points = Float64[]
    carbon_points = Float64[]
    
    for fixed_weight in fixed_weights
        SDDP.set_trade_off_weight(model, fixed_weight)
        local simulations = SDDP.simulate(model, 1000, [:q, :b, :x, :f, :s, :ω_demand])
        # Print number of simulation trajectories for this weight.
        # @printf("For fixed weight %.2f, number of simulation trajectories: %d\n", fixed_weight, length(simulations))
        local avg_cost, avg_carbon = average_cost_and_carbon(simulations, products, resources, T, 
            product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
        push!(fixed_weight_points, fixed_weight)
        push!(cost_points, avg_cost)
        push!(carbon_points, avg_carbon)
        @printf("Weight: %.2f  Avg Cost: %.2f, Avg Carbon: %.2f\n", fixed_weight, avg_cost, avg_carbon)
    end
    return fixed_weight_points, cost_points, carbon_points
end

# Define a range of fixed weights.
fixed_weights = 0.05:0.05:1.0

# Compute the averaged Pareto front points.
fw_pts, cost_pts, carbon_pts = compute_pareto_points_3d(model, fixed_weights, products, resources, T,
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)

# ---------------------------
# Create an Interactive 3D Scatter Plot using PlotlyJS via Plots.
# ---------------------------
# Use explicit qualification to call Plots.scatter3d.
p3d = Plots.scatter3d(fw_pts, cost_pts, carbon_pts,
    xlabel="Fixed Weight",
    ylabel="Total Cost",
    zlabel="Total Carbon Emissions",
    title="Interactive 3D Pareto Front (Averaged over 100 replications)",
    marker=(:circle, 1.5),
    legend=false,
    size=(900,700))

#display(p3d)


# 计算每个固定权重下的加权目标值
weighted_obj = [(1 - fw_pts[i]) * cost_pts[i] + fw_pts[i] * carbon_pts[i] for i in 1:length(fw_pts)]

# 绘制图1：固定权重 vs. 加权目标值
p1 = Plots.plot(fw_pts, weighted_obj,
    xlabel="Fixed Weight",
    ylabel="Weighted Objective",
    marker=:circle,
    legend=false)

# 绘制图2：平均碳排放 vs. 平均成本
p2 = Plots.plot(carbon_pts, cost_pts,
    xlabel="Total Carbon Emissions",
    ylabel="Total Cost",
    marker=:circle,
    legend=false)

display(p1)
display(p2)

Plots.savefig(p1, "model4_weights_objectives_fixed.png")
Plots.savefig(p2, "model4_avg_cost_carbon_fixed.png")

# ---------------------------
# 过滤固定权重 > 0.1 的数据
# ---------------------------
indices = findall(x -> x > -1, fw_pts)
fw_pts_filtered = fw_pts[indices]
cost_pts_filtered = cost_pts[indices]
carbon_pts_filtered = carbon_pts[indices]

# 计算每个固定权重下的加权目标值（使用过滤后的数据）
weighted_obj = [(1 - fw_pts_filtered[i]) * cost_pts_filtered[i] + fw_pts_filtered[i] * carbon_pts_filtered[i]
                for i in 1:length(fw_pts_filtered)]

# 绘制图1：固定权重 vs. 加权目标值（过滤后的数据）
p1 = Plots.plot(fw_pts_filtered, weighted_obj,
    xlabel="Fixed Weight",
    ylabel="Weighted Objective",
    marker=:circle,
    legend=false)

# 绘制图2：平均碳排放 vs. 平均成本（过滤后的数据）
p2 = Plots.plot(carbon_pts_filtered, cost_pts_filtered,
    xlabel="Total Carbon Emissions",
    ylabel="Total Cost",
    marker=:circle,
    legend=false)

display(p1)
display(p2)

Plots.savefig(p1, "model4_weights_objectives_fixed_filtered.png")
Plots.savefig(p2, "model4_avg_cost_carbon_fixed_filtered.png")
