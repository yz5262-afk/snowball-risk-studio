from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.dpi':150, 'axes.titleweight':'bold'})
(ROOT/'figures').mkdir(exist_ok=True)

import numpy as np
from snowball_engine import SnowballParams, generate_normals, greeks_curve, barrier_schedule
p=SnowballParams(n_sims=10000)
p.validate()
z=generate_normals(p,seed=42)
curve=greeks_curve(np.linspace(40,130,60),0.0,z,p)
barriers=barrier_schedule(p)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for ax,column in zip(axes[:2],['Price','Delta']):
    ax.plot(curve.Spot,curve[column],color='#157a83',lw=2)
    ax.axvline(p.KI_barrier,color='#d69b3a',ls='--',lw=1)
    ax.set(xlabel='Underlying spot',ylabel=column,title=f'Monte Carlo {column}')
axes[2].step(barriers.Observation_Month,barriers.KO_Barrier,color='#d69b3a',where='mid')
axes[2].set(xlabel='Observation month',ylabel='KO barrier',title='Step-down schedule')
fig.suptitle('Snowball Risk Studio | synthetic inputs, 10,000 paths',fontsize=12)
fig.savefig(ROOT/'figures/model-analytics.svg')
curve.to_csv(ROOT/'figures/model-analytics.csv',index=False)
print('Generated model-analytics.svg and model-analytics.csv')
