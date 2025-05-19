using SDDP, HiGHS, Random, Printf, Gurobi, Test, Statistics, Plots


T = 12
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

# Resource capacity
resource_capacity = Dict("A" => 350, "B" => 350, "C" => 350)

# Carbon cap per period
carbon_cap = [1000000 for t in 1:T]  

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

# 🌟 优化模型（带随机需求）
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

        # 随机需求扰动
        Ω = [-0.25, -0.1 ,0, 0.1, 0.25]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for product in products
                fix(ω_demand[product], demand_data[product][t] * (1 + ω))
            end

            # 定义双目标函数
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

    # 梯度测试（验证双目标曲线的单调性）
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

# 运行模型
model, solutions_summary = biobjective_product_production()

# ---------------------------
# 模拟评估辅助函数
# ---------------------------
# 计算单个模拟轨迹的总成本和总碳排放
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

# 平均所有模拟结果的成本和碳排放
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
# 计算基于模拟的 Pareto 前沿点（3D均值）并直接打印加权目标值
# ---------------------------
function compute_pareto_points_3d(model, fixed_weights, products, resources, T,
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
    
    fixed_weight_points = Float64[]
    cost_points = Float64[]
    carbon_points = Float64[]
    
    for fixed_weight in fixed_weights
        SDDP.set_trade_off_weight(model, fixed_weight)
        local simulations = SDDP.simulate(model, 1000, [:q, :b, :x, :f, :s, :ω_demand])
        local avg_cost, avg_carbon = average_cost_and_carbon(simulations, products, resources, T, 
            product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
        push!(fixed_weight_points, fixed_weight)
        push!(cost_points, avg_cost)
        push!(carbon_points, avg_carbon)
        # 计算加权目标值
        local weighted_obj = fixed_weight * avg_cost + (1 - fixed_weight)  * avg_carbon
        @printf("Weight: %.5f  Avg Cost: %.2f, Avg Carbon: %.2f, Weighted Objective: %.2f\n",
                fixed_weight, avg_cost, avg_carbon, weighted_obj)
    end
    return fixed_weight_points, cost_points, carbon_points
end

# ---------------------------
# 使用训练得到的推荐权重进行模拟评估
# ---------------------------
# 从训练返回的 solutions_summary 中提取推荐的权重
recommended_weights = [w for (w, _) in solutions_summary]

# 用推荐的权重计算对应的模拟结果，并直接打印加权目标值
fw_pts_rec, cost_pts_rec, carbon_pts_rec = compute_pareto_points_3d(
    model,
    recommended_weights,
    products, resources, T,
    product_resource_params,
    inventory_costs, inventory_carbon,
    waste_costs, backorder_costs
)

# ---------------------------
# 绘图部分
# ---------------------------
# 计算每个推荐权重下的加权目标值
weighted_obj = [(1 - fw_pts_rec[i]) * cost_pts_rec[i] + fw_pts_rec[i] * carbon_pts_rec[i] 
                for i in 1:length(fw_pts_rec)]

# 绘制图1：权重 vs. 加权目标值
p1 = Plots.plot(fw_pts_rec, weighted_obj,
    xlabel="Weight",
    ylabel="Weighted Objective",
    marker=:circle,
    legend=false)

# 绘制图2：平均成本 vs. 平均碳排放
p2 = Plots.plot(carbon_pts_rec, cost_pts_rec,
    xlabel="Avg Carbon",
    ylabel="Avg Cost",
    marker=:circle,
    legend=false)

# 显示绘图结果
display(p1)
display(p2)

Plots.savefig(p1, "model4_weights_objectives.png")
Plots.savefig(p2, "model4_avg_cost_carbon.png")

# ---------------------------
# 过滤推荐权重 > 0.1 的数据
# ---------------------------
indices = findall(x -> x > -1, fw_pts_rec)
fw_pts_rec_filtered = fw_pts_rec[indices]
cost_pts_rec_filtered = cost_pts_rec[indices]
carbon_pts_rec_filtered = carbon_pts_rec[indices]

# 计算过滤后的加权目标值
weighted_obj_filtered = [(1 - fw_pts_rec_filtered[i]) * cost_pts_rec_filtered[i] +
                           fw_pts_rec_filtered[i] * carbon_pts_rec_filtered[i]
                           for i in 1:length(fw_pts_rec_filtered)]

# ---------------------------
# 绘图部分（过滤后的数据）
# ---------------------------
# 绘制图1：权重 vs. 加权目标值
p1 = Plots.plot(fw_pts_rec_filtered, weighted_obj_filtered,
    xlabel="Weight",
    ylabel="Weighted Objective",
    marker=:circle,
    legend=false)

# 绘制图2：平均碳排放 vs. 平均成本
p2 = Plots.plot(carbon_pts_rec_filtered, cost_pts_rec_filtered,
    xlabel="Avg Carbon",
    ylabel="Avg Cost",
    marker=:circle,
    legend=false)

# 显示绘图结果
display(p1)
display(p2)

# 保存绘图
Plots.savefig(p1, "model4_weights_objectives_heuristics_filtered.png")
Plots.savefig(p2, "model4_avg_cost_carbon_heuristics_filtered.png")
