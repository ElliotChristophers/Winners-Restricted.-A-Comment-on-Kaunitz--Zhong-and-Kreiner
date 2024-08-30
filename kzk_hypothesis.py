import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from scipy.stats import wasserstein_distance,ttest_1samp
from scipy.optimize import minimize, LinearConstraint
import random
from tqdm import tqdm
import itertools

#Metrics for Kaunitz et al
profit = 957.5
acc = 0.472
n_bets = 265
stake = 50
n_bet_bookmaker_range = [4,32]

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
        

fig = plt.figure()
for i in range(9):
    ax = fig.add_subplot(331+i)
    ax.hist(generate_odds(),bins=10,range=(1,6),density=True,color='0.8', edgecolor='black')
    ax.set_xlabel(r"$o_k'$",fontsize=13)
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

significance_level_range = [0.01,0.1]

l = []
for m_tilde, p_tilde, alpha_tilde, tau in tqdm(list(itertools.product([0,1],[0,1],[0,1],[5,15,30]))):
    restrictions = []
    for i in tqdm(range(1000)):
        gen_returns = list(generate_odds()) + [-1 for _ in range(n_losses)]
        
        if not m_tilde:
            n_bet_bookmakers = n_bet_bookmaker_range[0]
        else:
            n_bet_bookmakers = random.randint(n_bet_bookmaker_range[0],n_bet_bookmaker_range[1])
        bookmaker_returns = [[] for _ in range(n_bet_bookmakers)]
        
        if not p_tilde:
            weights = np.ones(n_bet_bookmakers) / n_bet_bookmakers
        else:
            weights = np.random.random(n_bet_bookmakers)
            weights /= np.sum(weights)
        
        for ret in gen_returns:
            chosen_bookmaker = random.choices(bookmaker_returns, weights=weights, k=1)[0]
            chosen_bookmaker.append(ret)

        if not alpha_tilde:
            sig_levs = 0.05*np.ones(n_bet_bookmakers)
        else:
            sig_levs = np.random.uniform(significance_level_range[0],significance_level_range[1],size=n_bet_bookmakers)
        
        p_vals = np.array([ttest_1samp(rets,0,alternative='greater')[1] if len(rets) > tau else 1 for rets in bookmaker_returns])
        restrictions.append(np.sum(np.where(p_vals < sig_levs,1,0)))

    l.append([m_tilde, p_tilde, alpha_tilde, tau, 1 - np.mean(np.array(restrictions)>=4)])

df = pd.DataFrame(l,columns=['m_tilde','p_tilde','a_tilde','tau','p_theta'])

df.sort_values(by='p_theta',ascending=True)

unique_pairs = []
for column in df.columns.values[:-1]:
    for value in df[column].unique():
        unique_pairs.append({'parameter':column, 'value':value,'bar_p_theta':df[df[column]==value]['p_theta'].mean()})
df = pd.DataFrame(unique_pairs)

df = pd.DataFrame(l,columns=['m_tilde','p_tilde','a_tilde','tau','p_theta'])
df.groupby(by=['m_tilde','p_tilde','a_tilde'],axis=0).median().drop(columns=['tau'])