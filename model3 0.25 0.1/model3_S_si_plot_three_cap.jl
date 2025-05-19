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
caps = [11000 for t in 1:T]

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
# 构建带随机需求与资源容量约束的SDDP模型
# ---------------------------
function stochastic_product_production(caps::Vector{<:Number})
    model = SDDP.LinearPolicyGraph(
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
    ) do sp, t
        # 状态变量：库存 (q) 和 缺货 (b)
        @variable(sp, q[products] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, b[products] >= 0, SDDP.State, initial_value = 0)
        # 决策变量：生产 (x)、库存销售 (s)、废弃 (f) 与需求扰动 (ω_demand)
        @variable(sp, x[products, resources] >= 0)
        @variable(sp, f[products] >= 0)
        @variable(sp, s[products] >= 0)
        @variable(sp, ω_demand[products])
        
        # 随机需求扰动：使用扰动集合 [-50, -20, 0, 20, 50] 及对应概率
        #Ω = [-0.5, -0.2 ,0, 0.2, 0.5]
        #Ω = [0,0,0,0,0]
        Ω = [-0.25, -0.1 ,0, 0.1, 0.25]
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

        # 资源生产能力约束：限制每个资源在每个阶段的总生产量
        for r in resources
            @constraint(sp, sum(x[product, r] for product in products) <= production_caps[r])
        end

        # 目标函数：最小化生产成本、库存成本、废弃成本和缺货成本
        @stageobjective(sp,
            sum(product_resource_params[product][r]["prod_cost"] * x[product, r] for product in products, r in resources) +
            sum(inventory_costs[product] * s[product] for product in products) +
            sum(waste_costs[product] * f[product] for product in products) +
            sum(backorder_costs[product] * b[product].out for product in products)
        )
    end

    SDDP.train(model; iteration_limit = 10)
    global last_lower_bound = SDDP.calculate_bound(model)
    println("Lower bound: ", last_lower_bound)
    return model  # 只返回 model，方便 simulate
end

# 运行模型（使用默认碳排放上限 caps_default）
model = stochastic_product_production(caps)

# ---------------------------
# Simulation & Visualization
# ---------------------------
number_simulations = 1000
simulations = SDDP.simulate(model, number_simulations, [:q, :b, :x, :f, :s, :ω_demand])

# 辅助函数：计算平均总成本与平均碳排放
function compute_average_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, 
    waste_costs, backorder_costs)

    total_costs = []
    total_carbons = []

    for sim in simulations
        total_cost = 0.0
        total_carbon = 0.0

        for t in 1:T
            for product in products, r in resources
                prod_val = sim[t][:x][product, r]
                total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
                total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
            end
            for product in products
                total_cost += inventory_costs[product] * sim[t][:s][product]
                total_carbon += inventory_carbon[product] * sim[t][:s][product]
                total_cost += waste_costs[product] * sim[t][:f][product]
                total_cost += backorder_costs[product] * sim[t][:b][product].out
            end
        end

        push!(total_costs, total_cost)
        push!(total_carbons, total_carbon)
    end

    return mean(total_costs), mean(total_carbons)
end

# 辅助函数：计算单个模拟轨迹
function compute_total_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, 
    waste_costs, backorder_costs)
    
    total_cost = 0.0
    total_carbon = 0.0
    sim = simulations[42]  # 选择第42个模拟轨迹

    for t in 1:T
        for product in products, r in resources
            prod_val = sim[t][:x][product, r]
            total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
            total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
        end
        for product in products
            total_cost += inventory_costs[product] * sim[t][:s][product]
            total_carbon += inventory_carbon[product] * sim[t][:s][product]
            total_cost += waste_costs[product] * sim[t][:f][product]
            total_cost += backorder_costs[product] * sim[t][:b][product].out
        end
    end

    return total_cost, total_carbon
end

tc, tcarbon = compute_total_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)

@printf("Under the optimal solution (42nd simulation), the total cost is: %.2f\n", tc)
@printf("Under the optimal solution (42nd simulation), the total carbon emissions are: %.2f\n", tcarbon)

avg_tc, avg_tcarbon = compute_average_cost_and_carbon(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, waste_costs, backorder_costs)

@printf("Average total cost over %d simulations: %.2f\n", number_simulations, avg_tc)
@printf("Average total carbon emissions over %d simulations: %.2f\n", number_simulations, avg_tcarbon)

using Plots, Statistics

function compute_all_costs_and_carbons(simulations, products, resources, T, 
    product_resource_params, inventory_costs, inventory_carbon, 
    waste_costs, backorder_costs)
    
    total_costs = Float64[]
    total_carbons = Float64[]
    
    for sim in simulations
        total_cost = 0.0
        total_carbon = 0.0
        for t in 1:T
            for product in products, r in resources
                prod_val = sim[t][:x][product, r]
                total_cost += product_resource_params[product][r]["prod_cost"] * prod_val
                total_carbon += product_resource_params[product][r]["carbon_emission"] * prod_val
            end
            for product in products
                total_cost += inventory_costs[product] * sim[t][:s][product]
                total_carbon += inventory_carbon[product] * sim[t][:s][product]
                total_cost += waste_costs[product] * sim[t][:f][product]
                total_cost += backorder_costs[product] * sim[t][:b][product].out
            end
        end
        push!(total_costs, total_cost)
        push!(total_carbons, total_carbon)
    end

    return total_costs, total_carbons
end


# ---------------------------
# 扩展情形：更多碳排放上限取值
# ---------------------------
using DataFrames, CSV, Printf

all_costs, all_carbons = compute_all_costs_and_carbons(
    simulations, products, resources, T,
    product_resource_params, inventory_costs, inventory_carbon,
    waste_costs, backorder_costs
)

# 当前的 cap 值
cap_value = caps[1]

# 构造 DataFrame，Cap 放在第一列
df = DataFrame(
    Cap = fill(cap_value, length(all_costs)),
    CarbonEmissions = all_carbons,
    TotalCost = all_costs
)

# 目标文件名（所有 cap 结果合并在一个文件中）
csv_filename = joinpath(@__DIR__, "model3_low_simulation_results_all_caps.csv")

# 判断是否已有文件：存在就追加，否则创建新文件
if isfile(csv_filename)
    existing_df = CSV.read(csv_filename, DataFrame)
    combined_df = vcat(existing_df, df)
    CSV.write(csv_filename, combined_df)
    println("Appended cap = $cap_value data to existing CSV.")
else
    CSV.write(csv_filename, df)
    println("Created new CSV with cap = $cap_value data.")
end



using DataFrames, CSV, Plots, Printf

# 读取已保存的 CSV 数据
csv_filename = joinpath(@__DIR__, "model3_low_simulation_results_all_caps.csv")
df_all = CSV.read(csv_filename, DataFrame)

# 获取唯一的 cap 值，用于分组绘图
unique_caps = sort(unique(df_all.Cap))
colors = distinguishable_colors(length(unique_caps))  # 自动生成可区分颜色

# 初始化图
p = plot(
    xlabel = "Total Carbon Emissions",
    ylabel = "Total Cost",
    #title = "Simulation Results by Carbon Cap",
    legend = :topright,
    grid = true,
    xformatter = x -> @sprintf("%d", round(Int, x)),
    yformatter = y -> @sprintf("%d", round(Int, y))
)

# 为每个 cap 分组绘图
for (i, cap) in enumerate(unique_caps)
    subdf = filter(row -> row.Cap == cap, df_all)
    scatter!(p,
        subdf.CarbonEmissions,
        subdf.TotalCost,
        label = "Cap = $(cap)",
        color = colors[i],
        markersize = 3,
        alpha = 0.6
    )
end

# 显示并保存图像
display(p)
plot_filename = joinpath(@__DIR__, "model3_low_all_caps_simulation_plot.png")
savefig(p, plot_filename)
println("Plot saved to: ", plot_filename)
