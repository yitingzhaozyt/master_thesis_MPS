using SDDP, HiGHS, Random, Printf, Gurobi, Plots, Statistics

# ---------------------------
# 数据定义
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
caps = [0 for t in 1:T]

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
# 构建带随机需求的SDDP模型
# ---------------------------
function stochastic_prod_production()
    model = SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # 状态变量：q 为库存（carryover），b 为缺货
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, x[products, resources] >= 0)
        @variable(sp, f[products] >= 0)
        @variable(sp, s[products] >= 0)
        @variable(sp, ω_demand[products])
        
        # 随机需求扰动：每个阶段的需求在初始需求上加上扰动（-50, -20, 0, 20, 50）
        Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        #Ω = [0,0,0,0,0]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for prod in products
                fix(ω_demand[prod], demand_data[prod][t] * (1 + ω))
            end
        end

        # 供需约束与状态更新
        for prod in products
            @constraint(sp, s[prod] >= sum(x[prod, r] for r in resources) - ω_demand[prod] + q[prod].in - b[prod].in)
            @constraint(sp, b[prod].out >= -sum(x[prod, r] for r in resources) + ω_demand[prod] - q[prod].in + b[prod].in)
            @constraint(sp, f[prod] == s[prod] * spoilage_rates[prod])
            @constraint(sp, q[prod].out == s[prod] - f[prod])
            @constraint(sp, ω_demand[prod] == -b[prod].in + q[prod].in + sum(x[prod, r] for r in resources) - s[prod] + b[prod].out)
        end

        # ADD: Production capacity constraints – for each resource, total production over all products ≤ capacity.
        for r in resources
            @constraint(sp, sum(x[prod, r] for prod in products) <= production_caps[r])
        end

        # 碳排放约束
        @constraint(sp,
            sum(product_resource_params[prod][r]["carbon_emission"] * x[prod, r] for prod in products, r in resources) +
            sum(inventory_carbon[prod] * s[prod] for prod in products) <= caps[t]
        )

        # 目标函数：最小化生产成本、库存成本、废弃成本和缺货成本
        @stageobjective(sp,
            sum(product_resource_params[prod][r]["prod_cost"] * x[prod, r] for prod in products, r in resources) +
            sum(inventory_costs[prod] * s[prod] for prod in products) +
            sum(waste_costs[prod] * f[prod] for prod in products) +
            sum(backorder_costs[prod] * b[prod].out for prod in products)
        )
    end

    # 训练模型；iteration_limit 可根据情况调整以改善下界
    SDDP.train(model; iteration_limit = 10)
    println("Lower bound: ", SDDP.calculate_bound(model))
    
    return model
end
 
# 运行优化并获取训练好的模型
model = stochastic_prod_production()
