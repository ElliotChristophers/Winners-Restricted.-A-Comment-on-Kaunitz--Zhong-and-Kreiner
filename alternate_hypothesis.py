import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance,rankdata, wilcoxon
from scipy.optimize import minimize, LinearConstraint
import random
from tqdm import tqdm
import itertools

#Metrics for Kaunitz et al
profit = 957.5
acc = 0.472
n_bets = 265
stake = 50
m_i_range = [4,32]

n_wins = int(acc*n_bets//1)
n_losses = n_bets - n_wins

bet_odds = np.genfromtxt('odds.csv')

def generate_odds():
    lb = ub = stake * n_bets + profit
    A = np.ones(n_wins) * stake
    profit_constraint = LinearConstraint(A, lb, ub)

    x0 = random.sample(list(bet_odds),n_wins)
    bounds = [(np.min(bet_odds),np.max(bet_odds)) for _ in range(len(x0))]
    return minimize(lambda x: wasserstein_distance(x,bet_odds),x0=x0,bounds=bounds,constraints=profit_constraint).x

m_hi_min = 10
mu_i_max = 100
alpha_range = [0.01,0.1]
delta_range = [0.1,0.5]
alpha_constant = 0.05
delta_constant = 0.25

l = []
for m_tilde, mh_tilde, p_tilde, alpha_tilde, delta_tilde in tqdm(list(itertools.product([0, 1], repeat=5))):
    restrictions = []
    for i in tqdm(range(1000)):
        gen_returns = sorted(list(generate_odds()) + [-1 for _ in range(n_losses)], key=lambda x: random.random())
        if not m_tilde:
            m_i = m_i_range[0]
        else:
            m_i = random.randint(m_i_range[0],m_i_range[1])

        mu_i = random.randint(m_i,mu_i_max)
        T_i = sorted(np.random.choice(mu_i,size=m_i,replace=False))

        if not mh_tilde:
            T_hi = [sorted([i]+list(np.random.choice([j for j in range(mu_i) if j!=i],size=min(m_i,m_hi_min)-1,replace=False))) for i in range(mu_i)]
        else:
            T_hi = [sorted([i]+list(np.random.choice([j for j in range(mu_i) if j!=i],size=random.randint(min(m_i,m_hi_min),mu_i)-1,replace=False))) for i in range(mu_i)]
        
        if not p_tilde:
            P_i = np.ones(mu_i) / mu_i
        else:
            P_i = np.random.random(mu_i)
            P_i /= np.sum(P_i)
        
        results = [[[],[]] for _ in range(mu_i)]

        for ret in gen_returns:
            Pp_i = np.random.random(mu_i)
            R_i = rankdata(P_i*Pp_i,method='max')
            i = T_i[np.argmax([R_i[i] for i in T_i])]
            R_ijp = rankdata([R_i[i] for i in T_hi[i]],method='max')[list(T_hi[i]).index(i)]
            results[i][0].append(R_ijp)
            results[i][1].append(ret)

        if not alpha_tilde:
            Alpha_i = alpha_constant*np.ones(mu_i)
        else:
            Alpha_i = np.random.uniform(alpha_range[0],alpha_range[1],size=mu_i)
        
        if not delta_tilde:
            Delta_i = delta_constant*np.ones(mu_i)
        else:
            Delta_i = np.random.uniform(delta_range[0],delta_range[1],size=mu_i)
        
        p_vals = np.array([wilcoxon(1e-5+np.array(results[i][0])-(1-Delta_i[i])*len(T_hi[i]),alternative='greater')[1] if (np.mean(results[i][1]) if len(results[i][1]) > 0 else -1) > 0 else 1 for i in range(mu_i)])
        restrictions.append(np.sum(np.where(p_vals < Alpha_i,1,0)))

    l.append([m_tilde, mh_tilde, p_tilde, alpha_tilde, delta_tilde, 1 - np.mean(np.array(restrictions)>=4)])

df = pd.DataFrame(l,columns=['m_tilde','mh_tilde','p_tilde','alpha_tilde','delta_tilde','p_theta'])

df.sort_values(by='p_theta',ascending=True)

unique_pairs = []
for column in df.columns.values[:-1]:
    for value in df[column].unique():
        unique_pairs.append({'parameter':column, 'value':value,'bar_p_theta':df[df[column]==value]['p_theta'].mean(),'min_p_theta':df[df[column]==value]['p_theta'].min()})
df = pd.DataFrame(unique_pairs)


df = pd.DataFrame(l,columns=['m_tilde','mh_tilde','p_tilde','alpha_tilde','delta_tilde','p_theta'])
df.groupby(by=['m_tilde','p_tilde','alpha_tilde'],axis=0).median().drop(columns=['mh_tilde','delta_tilde'])