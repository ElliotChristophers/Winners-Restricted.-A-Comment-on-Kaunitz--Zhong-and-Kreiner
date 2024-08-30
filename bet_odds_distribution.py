import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from scipy.stats import ttest_1samp


list_betting_firms = ['B365', 'BW', 'IW', 'PS', 'VC', 'WH']
venues = ['H','D','A']

alpha = 0.05

df = pd.read_csv('all_data.csv',index_col=0)
df['n_bookmakers'] = 13-df.isna().sum(axis=1)/3

len(df) - df.isna().sum(axis=0)

for v in venues:
    df[f'{v}_longest'] = df[[i+v for i in list_betting_firms]].max(axis=1,skipna=True).values
    df[f'{v}_mean'] = df[[i+v for i in list_betting_firms]].mean(axis=1,skipna=True).values
    df[f'{v}_p'] = 1/df[f'{v}_mean'] - alpha
    df[f'{v}_ev'] = df[f'{v}_p'] * df[f'{v}_longest'] - 1
df['H_return_longest'] = np.where(df['FTHG']>df['FTAG'],df['H_longest']-1,-1)
df['D_return_longest'] = np.where(df['FTHG']==df['FTAG'],df['D_longest']-1,-1)
df['A_return_longest'] = np.where(df['FTHG']<df['FTAG'],df['A_longest']-1,-1)
df['bet'] = 1*(df[['H_ev','D_ev','A_ev']].max(axis=1) > 0)

df_bet = df[df['bet']==1].drop(columns=['bet']).reset_index(drop=True)
df_bet['bet_idx'] = [np.argmax(df_bet.loc[i,['H_ev','D_ev','A_ev']]) for i in df_bet.index]
df_bet['return'] = [(df_bet.loc[i,[f'{v}_return_longest' for v in venues][j]]) for i,j in enumerate(df_bet['bet_idx'])]
df_bet['odds'] = [(df_bet.loc[i,[f'{v}_longest' for v in venues][j]]) for i,j in enumerate(df_bet['bet_idx'])]

profit = df_bet['return'].sum()

df_bet['odds'].to_csv('odds.csv',index=None,header=None)

progression = [0] + list(np.cumsum(df_bet['return']))
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(range(len(progression)),progression,color='0.2',linewidth=0.5)
ax.set_xlabel(r'$k$',fontsize=13)
ax.set_ylabel(r'$\pi_k$',fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = fig.add_subplot(212)
ax.hist(df_bet['odds'],bins=10,range=(1,6),density=True,color='0.8', edgecolor='black')
ax.set_xlabel(r'$o_k$',fontsize=13)
ax.set_ylabel('Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print(profit,len(df_bet),len(df))
print(ttest_1samp(df_bet['return'],0,alternative='greater')[1])
