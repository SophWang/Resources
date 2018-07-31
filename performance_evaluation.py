import matplotlib.pyplot as plt
import numpy as np
from empyrical import max_drawdown, alpha_beta


class Evaluation(object):
    def __init__(self, prices, n, dps, step, threshold, benchmark_returns, len_of_benchmark):
        """
        prices: 价格数据
        n: 窗口长度, an integer
        dps: 平均价格变化
        step: 交易时间间隔
        thresohld: 进行交易的价格界限
        benchmark_returns: 市场平均回报率
        """
        self.prices = prices
        self.n = n
        self.dps = dps
        self.step = step
        self.threshold = threshold
        self.benchmark_returns = benchmark_returns
        self.len_of_benchmark = len_of_benchmark

    """
    建立虚拟账户并根据交易策略进行买卖
    返回最终账户余额和总投资金额
    """

    def visual_account(self):
        bank_balance = 0
        position = 0
        cost = 0
        step_i = self.step
        buy_hist = []
        sell_hist = []
        hist_balance = np.random.randn(self.len_of_benchmark, 1)
        hist_cost = np.random.randn(self.len_of_benchmark, 1)
        for i in range(self.n, len(self.prices[0]) - 1, step_i):
            if i <= (self.n + self.len_of_benchmark - 1):
                hist_cost[i - self.n, 0] = self.prices[0, i]
            if self.dps[i - self.n] > self.threshold and position < 0:
                position += 1
                bank_balance -= self.prices[0, i]
                cost += self.prices[0, i]
                buy_hist.append(self.prices[0, i])
            if self.dps[i - self.n] < -self.threshold and position >= 0:
                position -= 1
                bank_balance += self.prices[0, i]
                sell_hist.append(self.prices[0, i])
            if i <= (self.n + self.len_of_benchmark - 1):
                hist_balance[i - self.n, 0] = bank_balance
        current_price = self.prices[0, len(self.prices[0]) - 1]
        if position == 1:
            bank_balance += current_price
            buy_hist.append(current_price)
        if position == -1:
            bank_balance -= current_price
            sell_hist.append(current_price)
        buy = np.array(buy_hist)
        sell = np.array(sell_hist)
        profit = sell - buy
        final_profit = np.random.randn(1, len(profit))
        final_profit[0] = profit[0]
        for k in range(1, len(profit)):
            final_profit[0, k] = final_profit[0, k - 1] + profit[k]
        print(final_profit)
        return bank_balance, cost, hist_balance, hist_cost

    """
    计算最大回撤率
    返回最大回撤率，实际收益和按照Beta系数计算的期望收益之间的差额，业绩评价基准收益的总体波动性
    """

    def calculate_max_drawdown(self):
        final_balance, cost, balance, initial_cost = self.visual_account()
        returns = balance / initial_cost
        alpha, beta = alpha_beta(returns, self.benchmark_returns)
        maxdrawdown = max_drawdown(returns)
        plt.scatter(initial_cost, balance)
        plt.xlabel('cost history')
        plt.ylabel('balance history')
        plt.grid = (True)
        plt.show()
        print("Balance: " + str(final_balance) + " Investment cost: " + str(cost))
        print('max drawdown = ' + str(maxdrawdown) + '; alpha = ' + str(alpha) + '; beta= ' + str(beta) + '.')
        return maxdrawdown, alpha, beta
