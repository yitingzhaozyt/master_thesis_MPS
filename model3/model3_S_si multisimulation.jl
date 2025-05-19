using SDDP, HiGHS, Random, Printf, Gurobi, Plots, Statistics

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
#caps = [12000 for t in 1:T]

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
# 构建带随机需求和资源容量约束的SDDP模型
# ---------------------------
function stochastic_product_production()
    model = SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # 状态变量：q 为库存（carryover），b 为缺货
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)
        # 决策变量：生产 x、销售 s、废弃 f、以及需求扰动 ω_demand
        @variable(sp, x[products, resources] >= 0)
        @variable(sp, f[products] >= 0)
        @variable(sp, s[products] >= 0)
        @variable(sp, ω_demand[products])
        
        # 随机需求扰动：每个阶段需求在基础需求上加上扰动（-50, 0, 50）
        Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        #Ω = [0,0,0,0,0]
        P = [0.2, 0.25, 0.1, 0.25, 0.2]
        SDDP.parameterize(sp, Ω, P) do ω
            for prod in products
                fix(ω_demand[prod], demand_data[prod][t] * (1 + ω))
            end
        end

        # 供需约束与状态更新
        for product in products
            @constraint(sp, s[product] >= sum(x[product, r] for r in resources) - ω_demand[product] + q[product].in - b[product].in)
            @constraint(sp, b[product].out >= -sum(x[product, r] for r in resources) + ω_demand[product] - q[product].in + b[product].in)
            @constraint(sp, f[product] == s[product] * spoilage_rates[product])
            @constraint(sp, q[product].out == s[product] - f[product])
            @constraint(sp, ω_demand[product] == -b[product].in + q[product].in + sum(x[product, r] for r in resources) - s[product] + b[product].out)
        end

        # 碳排放约束
        @constraint(sp,
            sum(product_resource_params[product][r]["carbon_emission"] * x[product, r] for product in products, r in resources) +
            sum(inventory_carbon[product] * s[product] for product in products) <= caps[t]
        )
        
        # 资源生产能力约束：限制每个资源每个阶段的总生产
        for r in resources
            @constraint(sp, sum(x[product, r] for product in products) <= resource_capacity[r])
        end

        # 目标函数：最小化生产成本、库存成本、废弃成本和缺货成本
        @stageobjective(sp,
            sum(product_resource_params[product][r]["prod_cost"] * x[product, r] for product in products, r in resources) +
            sum(inventory_costs[product] * s[product] for product in products) +
            sum(waste_costs[product] * f[product] for product in products) +
            sum(backorder_costs[product] * b[product].out for product in products)
        )
    end

    # 训练模型；iteration_limit 可调整以改善下界
    SDDP.train(model; iteration_limit = 10)
    println("Lower bound: ", SDDP.calculate_bound(model))
    return model
end

# 运行优化并获取训练好的模型
model = stochastic_product_production()

# ---------------------------
# Simulation & Visualization
# ---------------------------
# 模拟100次，并收集所有状态和决策变量
simulations = SDDP.simulate(model, 100, [:q, :b, :x, :f, :s, :ω_demand])

using Statistics, Printf, Random

# ---------------------------
# 辅助函数
# ---------------------------
# 计算总成本（用于置信区间CI的计算）
function compute_total_cost(sim, products, resources, T, product_resource_params, inventory_costs, waste_costs, backorder_costs)
    total_cost = 0.0
    for t in 1:T
        # 生产成本
        for product in products, r in resources
            prod_val = sim[t][:x][product, r]
            total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
        end
        # 库存、废弃、和缺货成本
        for product in products
            total_cost += inventory_costs[product] * sim[t][:s][product]
            total_cost += waste_costs[product] * sim[t][:f][product]
            total_cost += backorder_costs[product] * sim[t][:b][product].out
        end
    end
    return total_cost
end

# 计算单个模拟轨迹的成本和碳排放
function compute_cost_and_carbon(sim, products, resources, T, product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
    total_cost = 0.0
    total_carbon = 0.0
    for t in 1:T
        # 生产部分：成本与碳排放
        for product in products, r in resources
            prod_val = sim[t][:x][product, r]
            total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
            total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
        end
        # 库存、废弃、和缺货部分：成本与碳排放
        for product in products
            total_cost += inventory_costs[product] * sim[t][:s][product]
            total_carbon += inventory_carbon[product] * sim[t][:s][product]
            total_cost += waste_costs[product] * sim[t][:f][product]
            total_cost += backorder_costs[product] * sim[t][:b][product].out
        end
    end
    return total_cost, total_carbon
end

# ---------------------------
# 主函数：显示文本输出并生成 LaTeX 表格
# ---------------------------
function display_simulations_with_table()
    # 计算所有模拟轨迹的目标值，用于置信区间CI的计算
    objective_values = [compute_total_cost(sim, products, resources, T, product_resource_params, inventory_costs, waste_costs, backorder_costs) for sim in simulations]
    
    # 计算SDDP置信区间（保留两位小数）
    μ, ci = round.(SDDP.confidence_interval(objective_values, 1.96); digits = 2)
    lower_bound = round(SDDP.calculate_bound(model); digits = 2)
    println("Confidence interval: ", μ, " ± ", ci)
    println("Lower bound: ", lower_bound)
    println()
    
    # 固定随机数种子保证结果复现
    Random.seed!(42)
    indices = sort(Random.shuffle(1:length(simulations))[1:10])
    
    # 准备文本输出与 LaTeX 表行
    text_outputs = ""
    latex_rows = ""
    
    for idx in indices
        cost, carbon = compute_cost_and_carbon(simulations[idx], products, resources, T, product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)
        # 构造文本输出
        text_line1 = @sprintf("Simulation %d: Cost and Carbon from simulation: %.2f and %.2f", idx, cost, carbon)
        inside_str = (cost >= (μ - ci) && cost <= (μ + ci)) ? "inside" : "outside"
        text_line2 = "The cost from simulation $(idx) is $(inside_str) the SDDP simulation confidence interval."
        text_outputs *= text_line1 * "\n" * text_line2 * "\n\n"
        
        # 构造 LaTeX 表格行
        latex_row = @sprintf("%d & %.2f & %.2f & %s \\\\ \n", idx, cost, carbon, (inside_str == "inside" ? "Yes" : "No"))
        latex_rows *= latex_row
    end
    
    # 打印文本输出
    println(text_outputs)
    
    # 生成 LaTeX 表格
    latex_table = """
    \\begin{table}[ht]
    \\centering
    \\begin{tabular}{cccc}
    \\hline
    Simulation & Cost & Carbon & Inside CI \\\\ \\hline
    """
    latex_table *= latex_rows
    latex_table *= """
    \\hline
    \\end{tabular}
    \\caption{Simulation outcomes for cost and carbon with CI comparison.}
    \\label{tab:simulation_results}
    \\end{table}
    """
    
    # 打印 LaTeX 表格
    println(latex_table)
end

# 调用函数显示模拟输出和 LaTeX 表格
display_simulations_with_table()
