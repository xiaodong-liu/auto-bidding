import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns


def sign(x, eps=1e-15):
    if (abs(x) < eps):
        return 0
    elif x > 0:
        return 1
    else:
        return -1
# 根据二分计算数值积分
def budget_constraint(budget, f, v_min, v_max, g, eps = 1e-15):
    '''
    :param budget: bidder's budget target
    :param f: bidder's value function
    :param v_min: bidder's value lower bound
    :param v_max: bidder's value upper bound
    :param g: the maximum bid formulated by other bidders except for bidder
    :return: the coefficient beta
    '''
    def helper(f, v_min, v_max, g, mid):
        h = lambda D, v : D * g(D) * f(v)
        return integrate.dblquad(h, v_min, v_max, lambda x : 0, lambda x : mid * x)
    l, r = 0.0, 1.0
    while(abs(l - r) > eps):
        mid = (l + r) / 2.0
        tmp = helper(f, v_min, v_max, g, mid)[0]
        if(abs(tmp - budget) < eps):
            return mid
        elif(tmp > budget):
            r = mid
        else:
            l = mid
    return r

# 利用二分求满足ROI的系数
def ROI_constraint(roi, f, v_min, v_max, g, eps = 1e-15):
    '''
    :param roi: bidder's ROI target
    :param f: bidder's value function
    :param v_min: bidder's value lower bound
    :param v_max: bidder's value upper bound
    :param g: the maximum bid formulated by other bidders except for bidder
    :return: the coefficient beta
    '''
    # 0 表示 满足 精度条件
    # 1 表示
    # 2 表示
    def helper(f, v_min, v_max, g, mid, roi, eps = 1e-15):
        h = lambda D, v: ((1 + roi) * D - v) * g(D) * f(v)
        tmp = integrate.dblquad(h, v_min, v_max, lambda x : 0, lambda x : mid * x)[0]
        if(abs(tmp) < eps):
            return 0
        elif(tmp > 0):
            return 1
        else:
            return 2

    b = lambda D, v : D * g(D) * f(v)
    a = integrate.dblquad(b, v_min, v_max, lambda x : 0, lambda x : x)[0]
    value = lambda D, v : v * g(D) * f(v)
    c = integrate.dblquad(value, v_min, v_max, lambda x : 0, lambda x : x)[0]
    roi_upper = c / a - 1
    if roi <= roi_upper:
        return 1
    else:
        l, r = 1 / (1 + roi), 1.0
        while(abs(l - r) > eps):
            mid = (r + l) / 2.0
            flag = helper(f, v_min, v_max, g, mid, roi, eps)
            if flag == 0:
                return mid
            elif flag == 1:
                r = mid
            else:
                l = mid
        return r


def get_coeff(data, v_min, v_max, f, g, is_ironing = False):
    '''
    :param data: pd.DataFrame (id, budget, roi)
    :param v_min:
    :param v_max:
    :param f:
    :param g:
    :param is_ironing:
    :return:
    '''
    if not is_ironing:
        def helper(x, v_min, v_max, f, g):
            beta_budget = budget_constraint(x.budget, f, v_min, v_max, g)
            beta_roi = ROI_constraint(x.roi, f, v_min, v_max, g)
            return (beta_budget, beta_roi, min(beta_budget, beta_roi))
        data[['beta_budget', 'beta_roi', 'beta']] = data.apply(lambda x:helper(x, v_min, v_max, f, g), axis = 1, result_type = 'expand')
        return data
    else:
        # 如果使用ironing
        _, ironing_points = ironing(f, v_min, v_max, g)
        def helper(x, v_min, v_max, f, g, ironing_points):
            # 找出上下两个边界点，然后根据边界点计算概率 (beta_1, beta_2, u)
            beta_budget = budget_constraint(x.budget, f, v_min, v_max, g)
            beta_roi = ROI_constraint(x.roi, f, v_min, v_max, g)
            beta = min(beta_budget, beta_roi)
            h = lambda D, v: D * g(D) * f(v)
            payment = integrate.dblquad(h, v_min, v_max, lambda x: 0, lambda x: beta * x)[0]
            l, r = 0, len(ironing_points)-1
            if(sign(payment-ironing_points[0][0]) <= 0):
                return (beta_budget, beta_roi, beta, 0, 1)
            elif(sign(payment-ironing_points[-1][0]) >= 0):
                return (beta_budget, beta_roi, 0, beta, 0)
            else:
                while(l < r):
                    mid = (l + r) // 2
                    if(sign(ironing_points[mid][0] - payment) >= 0):
                        r = mid
                    else:
                        l = mid + 1
                budget_1 = ironing_points[r-1][0]
                budget_2 = ironing_points[r][0]
                beta_1 = budget_constraint(budget_1, f, v_min, v_max, g)
                beta_2 = budget_constraint(budget_2, f, v_min, v_max, g)
                return (beta_budget, beta_roi, beta_1, beta_2, (payment - budget_2) / (budget_1 - budget_2))
        data[['beta_budget', 'beta_roi', 'beta_1', 'beta_2', 'prob']] = data.apply(lambda x : helper(x, v_min, v_max, f, g, ironing_points), axis = 1, result_type='expand')
        return data



def sample(budget_min, budget_max, roi_min, roi_max, nums):
    '''
    :param budget_min: [budget_min, budget_max]
    :param budget_max:
    :param roi_min: [roi_min, roi_max]
    :param roi_max:
    :return: (id, budget, roi) # 直接把参数计算出来
    '''
    #如果使用ironing
    data = pd.DataFrame(np.arange(nums), columns = ['id'])
    data[['budget', 'roi']] = data.apply(lambda x : (np.random.uniform(budget_min, budget_max), np.random.uniform(roi_min, roi_max)), axis = 1, result_type='expand')
    return data



# 根据所有的点对关系，返回上半部分凸包
# Andrew 算法寻找上凸壳

def find_curve(points):
    # 计算点a, b的叉乘, 叉乘大于0表示向量ac在向量ab的右边，等于零表示共线，小于0表示ac在ab左边
    def cross(a, b, c):
        x = (b[0] - a[0], b[1] - a[1])
        y = (c[0] - a[0], c[1] - a[1])
        return x[0] * y[1] - x[1] * y[0]
    points = sorted(points)
    res = []
    for point in points:
        while(len(res) >= 2 and sign(cross(res[-2], res[-1], point)) >= 0):
            res.pop()
        res.append(point)
    return res

def ironing(f, v_min, v_max, g):
    def helper(beta, f, v_min, v_max, g):
        # 计算给定beta下的payment，以及roi
        h = lambda D, v: D * g(D) * f(v)
        l = lambda D, v: v * g(D) * f(v)
        payment = integrate.dblquad(h, v_min, v_max, lambda x : 0, lambda x : beta * x)[0]
        value = integrate.dblquad(l, v_min, v_max, lambda x : 0, lambda x : beta * x)[0]
        return [payment, value]
    #1. 计算原来的ROI, budget曲线
    x = np.linspace(0.001, 1, 100)
    points = pd.Series(x).apply(lambda x : helper(x, f, v_min, v_max, g)).values.tolist()
    #2. 计算凸包，并且计算ironing后的曲线
    ironing_points = find_curve(points)
    return points, ironing_points

def test_budget():
    def f(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def g(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    v_min, v_max = 0, 1
    def check(budget):
        return np.sqrt(6 * budget)
    for b in np.linspace(0, 1, 10):
        beta_real = min(1.0, check(b))
        beta_test = budget_constraint(b, f, v_min, v_max, g)
        assert abs(beta_real - beta_test) < 1e-6

def test_roi():
    def f(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def g(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def check(roi):
        return 2.0 / (1 + roi)
    v_min, v_max = 0, 1
    for roi in np.linspace(1, 5, 10):
        roi_real = min(1, check(roi))
        roi_test = ROI_constraint(roi, f, v_min, v_max, g)
        assert abs(roi_real - roi_test) < 1e-5

def test_find_curve():
    points = np.random.random((10, 2)).tolist()
    res = find_curve(points)
    points = np.array(points)
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    print("------------")
    res = np.array(res)
    plt.show()
    plt.figure()
    plt.scatter(res[:, 0], res[:, 1])
    plt.show()

def test_ironing():
    def f(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def g(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    points, ironing_points = ironing(f, 0, 1, g)
    points = pd.DataFrame(points, columns=['x', 'y'])
    # ironing_points = pd.DataFrame(ironing_points, columns=['x', 'y'])
    sns.lineplot(data=points, x='x', y='y')
    # sns.lineplot(data=ironing_points, x='x', y='y')
    plt.show()

def test_sample():
    def f(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def g(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    budget_min, budget_max = 0.1, 1
    roi_min, roi_max = 0, 20
    data = sample(budget_min, budget_max, roi_min, roi_max, 5)
    print(data)

def test_get_coeff():
    def f(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def g(x):
        if x >= 0 and x <= 1:
            return 1
        else:
            return 0
    def helper(x, v_min, v_max, f, g):
        h = lambda D, v: D * g(D) * f(v)
        l = lambda D, v: v * g(D) * f(v)
        beta_1 = x.beta_1
        payment_1 = integrate.dblquad(h, v_min, v_max, lambda x: 0, lambda x: beta_1 * x)[0]
        value_1 = integrate.dblquad(l, v_min, v_max, lambda x: 0, lambda x: beta_1 * x)[0]
        beta_2 = x.beta_2
        payment_2 = integrate.dblquad(h, v_min, v_max, lambda x: 0, lambda x: beta_2 * x)[0]
        value_2 = integrate.dblquad(l, v_min, v_max, lambda x: 0, lambda x: beta_2 * x)[0]
        payment = x.prob * payment_1 + (1 - x.prob) * payment_2
        roi_1 = value_1 / (payment_1) - 1
        value = x.prob * value_1 + (1 - x.prob) * value_2
        roi_2 = value_2 / (payment_2
                           ) - 1
        roi = x.prob * roi_1 + (1 - x.prob) * roi_2
        # beta_t = min(x.beta_budget, x.beta_roi)
        # payment_t = integrate.dblquad(h, v_min, v_max, lambda x: 0, lambda x: beta_t * x)[0]
        # value_t = integrate.dblquad(l, v_min, v_max, lambda x: 0, lambda x: beta_2 * x)[0]
        return [payment, roi]
    budget_min, budget_max = 0.1, 1
    roi_min, roi_max = 0, 20
    data = sample(budget_min, budget_max, roi_min, roi_max, 5)
    data = get_coeff(data, 0, 1, f, g, is_ironing=True)
    points, ironing_points = ironing(f, 0, 1, g)
    points = pd.DataFrame(points, columns=['x', 'y'])
    ironing_points = pd.DataFrame(ironing_points, columns=['x', 'y'])
    sns.lineplot(data=points, x='x', y='y')
    sns.lineplot(data=ironing_points, x='x', y='y')
    data[['actual_payment', 'actual_roi']] = data.apply(lambda x : helper(x, 0, 1, f, g), axis=1, result_type='expand')
    print(data)
    sns.scatterplot(data=data, x='actual_payment', y='actual_roi')
    plt.show()



if __name__ == "__main__":
    test_budget()
    print("Budget OK")
    test_roi()
    print("ROI OK")
    test_ironing()
    test_get_coeff()