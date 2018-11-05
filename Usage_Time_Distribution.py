#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
构建不同面额优联票的使用时间分布
暂定两种面额：
40元、60元
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd

# 时间长度
T = 30
# 优惠券日使用比例
p1 = 0.5
p2 = 0.5
# 结算折扣
P_dis = 0.93
# 消耗速度分布的模拟次数
N = 20000

# 模拟：票的消耗速度分布
def gen_gamma_num(x):
    ga = np.random.gamma(1,x,N)
    cd = {}.fromkeys(range(2*T),0)
    for k,g in groupby(sorted(ga), key=lambda x: x//1):
        cd[k]=len(list(g))
    for k in list(cd.keys()):
        if k > T-1:
            cd[T-1] += cd.pop(k)
    return np.array(list(cd.values()))/N


# 模拟：票的日销量
def gen_daily_sell(t):
    x = np.arange(t)
    return -np.sin(0.91*x+0.8)*(x-20)*(x-20)*30-40*np.cos(0.33*x+0.2)*(x-20)*(x-20)+200*np.sqrt(x)+23*(x-30)**2+3500
#    return np.ones(t)*10800

# 作图：票的消耗速度分布
def plot_usage_speed():
    plt.figure(1)
    _coupon40 = gen_gamma_num(3)
    _coupon60 = gen_gamma_num(4)
    plt.plot(_coupon40,label='Coupon-40')
    plt.plot(_coupon60,label='Coupon-60')
    plt.xlabel('Days')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('FG1: Usage speed ditribution')


def cal_KOL_cost(s):
    '''
    拉一个新人，人气+2
    购票量增加200元，人气+1
    用票量增加100元，人气+1
    退票量增加200元，人气-1
    人气奖章奖励：100分一个区间，达到设定值即可获得奖章的奖励：
    0~4500的奖章: 每个有80元
    4600~9900: 100元
    10000~20000: 200元
    '''
    if s<4599:
        return s//100*80
    if s<9999:
        return (s-4500)//100*100+3600
    return (s-9900)//200 + 9000


# 计算：给KOL累计补贴成本
def cal_allowance(sell,usage):
    c = usage.cumsum()//100+np.append(sell,np.zeros(T)).cumsum()//200
    return np.array(list(map(cal_KOL_cost,c)))


def main():
    coupon40 = gen_gamma_num(3)
    coupon60 = gen_gamma_num(4)
    daily_sell = gen_daily_sell(T)
    daily_usage = np.zeros(2*T)
    for i in range(T):
        daily_usage[i:i+T] += (daily_sell[i]*p1*1.2*coupon40+daily_sell[i]*p2*1.2*coupon60)*P_dis
    acc_cost = (daily_usage-np.append(daily_sell,np.zeros(T))).cumsum()
    allow_cost = cal_allowance(daily_sell,daily_usage)
    total_cost = acc_cost+allow_cost
    plot_usage_speed()

    rng = pd.date_range('2018-3-1',periods=2*T)
    df = pd.DataFrame({'Daily sell':np.append(daily_sell,np.zeros(T)),
    'Daily usage':daily_usage,
    'Total cost':total_cost,
    'KOL cost':allow_cost},index=rng)
    df.round(2).to_csv('data.csv')

    df.plot(color=['b','r','k','y'])
    plt.axhline(0,lw=0.7,color='k')
    plt.axvline(df['Total cost'].idxmin(),lw=0.5,color='b')
    plt.title('FG3: Cost Analysis')
    plt.show()


if __name__ == "__main__":
    main()
