using JuMP, Gurobi, Random, DataFrames, CSV, Plots

# 基础数据
T = 12

Random.seed!(42)
products = ["product1", "product2", "product3", "product4", "product5"]
resources = ["A", "B", "C"]

# 生成需求数据
demand_data = Dict(prod => [rand(100:250) for _ in 1:T] for prod in products)

# 生产资源参数
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

# 其他成本和参数
inventory_costs = Dict("product1" => 1.6, "product2" => 2.06, "product3" => 2.67, "product4" => 1.93, "product5" => 1.23)
inventory_carbon = Dict("product1" => 1.172, "product2" => 0.507, "product3" => 1.69, "product4" => 1.924, "product5" => 0.904)
backorder_costs = Dict("product1" => 9.79, "product2" => 14.15, "product3" => 10.23, "product4" => 12.37, "product5" => 12.5)
waste_costs = Dict("product1" => 0.53, "product2" => 0.27, "product3" => 0.82, "product4" => 0.32, "product5" => 0.14)
spoilage_rates = Dict("product1" => 0.099, "product2" => 0.126, "product3" => 0.131, "product4" => 0.112, "product5" => 0.111)

# 资源生产能力
production_caps = Dict("A" => 350, "B" => 350, "C" => 350)

# 定义模型（传入lambda）
function deterministic_product_production(lambda)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, s[products, 1:T] >= 0)
    @variable(model, b[products, 1:T] >= 0)
    @variable(model, x[products, resources, 1:T] >= 0)
    @variable(model, f[products, 1:T] >= 0)

    # 初始阶段约束
    for prod in products
        @constraint(model, s[prod, 1] >= sum(x[prod, r, 1] for r in resources) - demand_data[prod][1])
        @constraint(model, b[prod, 1] >= -sum(x[prod, r, 1] for r in resources) + demand_data[prod][1])
        @constraint(model, demand_data[prod][1] == -s[prod, 1] + sum(x[prod, r, 1] for r in resources) + b[prod,1])
    end

    # 连续阶段约束
    for prod in products, t in 2:T
        @constraint(model, s[prod, t] >= (s[prod, t-1] + sum(x[prod, r, t] for r in resources)) - (f[prod, t-1] + demand_data[prod][t] + b[prod, t-1]))
        @constraint(model, b[prod, t] >= -(s[prod, t-1] + sum(x[prod, r, t] for r in resources)) + (f[prod, t-1] + demand_data[prod][t] + b[prod, t-1]))
        @constraint(model, demand_data[prod][t] == s[prod, t-1] + sum(x[prod, r, t] for r in resources) - f[prod, t-1] - s[prod, t] - b[prod, t-1] + b[prod, t])
    end

    # 损耗
    for prod in products, t in 1:T
        @constraint(model, f[prod, t] == s[prod, t] * spoilage_rates[prod])
    end

    # 资源产能限制
    for r in resources, t in 1:T
        @constraint(model, sum(x[prod, r, t] for prod in products) <= production_caps[r])
    end

    # 表达式
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

    # 目标函数
    @objective(model, Min, lambda * total_cost + (1 - lambda) * total_carbon)

    optimize!(model)

    return JuMP.value(total_cost), JuMP.value(total_carbon), JuMP.objective_value(model)
end

# ==== 主程序：循环不同lambda，收集结果 ====

lambda_values = 0:0.01:1
results = DataFrame(Lambda=Float64[], Total_Cost=Float64[], Total_Carbon=Float64[], Objective_Value=Float64[])

for lambda in lambda_values
    total_cost, total_carbon, objective_value = deterministic_product_production(lambda)
    push!(results, (lambda, total_cost, total_carbon, objective_value))
end

# 保存结果
CSV.write(joinpath(@__DIR__, "model2_bi_objective_results.csv"), results)

# ==== Plot 1：Lambda vs Total Cost 和 Total Carbon ====

plot1 = plot(results.Lambda, results.Total_Cost,
    label="Total Cost", xlabel="Lambda", ylabel="Value",
    size=(900,600), formatter=:plain, xtickfontsize=11, ytickfontsize=11, legendfontsize=11)
plot!(plot1, results.Lambda, results.Total_Carbon, label="Total Carbon")
savefig(plot1, joinpath(@__DIR__, "model2_tradeoff_totalcost_totalcarbon.png"))
display(plot1)

# 在 plot1 上标注 λ=0.0, 0.5, 1.0
for λ_target in [0.0, 0.5, 1.0]
    idx = findall(x -> isapprox(x, λ_target; atol=1e-6), results.Lambda)[1]
    annotate!(plot1, (results.Lambda[idx], results.Total_Cost[idx]), text("λ=$(round(λ_target, digits=1))", :black, :left, 8))
end

savefig(plot1, joinpath(@__DIR__, "model2_tradeoff_totalcost_totalcarbon_annotated.png"))


# ==== Plot 2：Pareto前沿 Total Carbon vs Total Cost ====

pareto_plot = plot(results.Total_Carbon, results.Total_Cost,
    xlabel="Total Carbon Emissions", ylabel="Total Cost",
    size=(900,600), formatter=:plain, xtickfontsize=11, ytickfontsize=11,
    label="Pareto Frontier", linewidth=2, color=:blue,
    markershape=:circle, markersize=3, legend=:topright)
savefig(pareto_plot, joinpath(@__DIR__, "model2_pareto_frontier.png"))
display(pareto_plot)


# ==== Plot 3：Normalized Cost vs Normalized Carbon ====

cost_min = minimum(results.Total_Cost)
cost_max = maximum(results.Total_Cost)
carbon_min = minimum(results.Total_Carbon)
carbon_max = maximum(results.Total_Carbon)

results[!, :Normalized_Cost] = (results.Total_Cost .- cost_min) ./ (cost_max - cost_min)
results[!, :Normalized_Carbon] = (results.Total_Carbon .- carbon_min) ./ (carbon_max - carbon_min)

normalized_plot = plot(results.Normalized_Carbon, results.Normalized_Cost,
    xlabel="Normalized Carbon", ylabel="Normalized Cost",
    label="Normalized Pareto Frontier",
    size=(900,600), formatter=:plain, xtickfontsize=11, ytickfontsize=11)
savefig(normalized_plot, joinpath(@__DIR__, "model2_normalized_pareto_frontier.png"))
display(normalized_plot)


# ==== Plot 4：Lambda vs Objective Value ====

objective_plot = plot(results.Lambda, results.Objective_Value,
    xlabel="Lambda", ylabel="Objective Value",
    size=(900,600), formatter=:plain, xtickfontsize=11, ytickfontsize=11,
    label="Objective Value", linewidth=2, color=:green,
    markershape=:circle, markersize=4)
savefig(objective_plot, joinpath(@__DIR__, "model2_lambda_vs_objective.png"))
display(objective_plot)


# ==== Plot 5：计算Marginal Rate (ΔCost / ΔCarbon) ====

n = nrow(results)
marginal_rate = Float64[]
lambda_mid = Float64[]

for i in 1:(n-1)
    Δcost = results.Normalized_Cost[i+1] - results.Normalized_Cost[i]
    Δcarbon = results.Normalized_Carbon[i+1] - results.Normalized_Carbon[i]
    if abs(Δcarbon) < 1e-6
        push!(marginal_rate, NaN)
    else
        push!(marginal_rate, Δcost / Δcarbon)
    end
    push!(lambda_mid, (results.Lambda[i+1] + results.Lambda[i]) / 2)
end

marginal_df = DataFrame(Lambda=lambda_mid, Marginal_Rate=marginal_rate)

marginal_plot = plot(marginal_df.Lambda, marginal_df.Marginal_Rate,
    xlabel="Lambda", ylabel="Marginal Rate (ΔNorm Cost / ΔNorm Carbon)",
    label="Marginal Rate",
    linewidth=2, markershape=:circle, markersize=4,
    size=(900,600), formatter=:plain, xtickfontsize=11, ytickfontsize=11)
savefig(marginal_plot, joinpath(@__DIR__, "model2_marginal_rate.png"))
display(marginal_plot)
