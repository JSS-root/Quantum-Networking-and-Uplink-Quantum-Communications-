from uplink_qkd import smart_optimise
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
# for different block size ratios sweep the qber difference between the blocks
m_total=1e6
avg_qber=0.05
m_ratio=np.linspace(0.2, 0.8, 4)
colours=['blue', 'orange', 'green', 'red']
qber1_range=np.linspace(0.02, 0.08, 100)
f, ax1  = plt.subplots(1, 1, figsize=(6, 4.5))
for i,r in tqdm(enumerate(m_ratio)):
    skl=[]
    m1 = r * m_total
    m2 = m_total - m1
    for qber1 in qber1_range:
        qber2 = (avg_qber - qber1*r)/(1-r)
        if qber2<0 or qber2>1:
            skl.append(np.nan)
            continue
        sk_ratio1 = smart_optimise(m1,qber1, 1e-6, np.log2(10**8) ,1.19, granularity=300)
        sk_ratio2 = smart_optimise(m2,qber2, 1e-6, np.log2(10**8) ,1.19, granularity=300)
        skl.append((sk_ratio1*m1 + sk_ratio2*m2)/m_total)

    ax1.plot(qber1_range*100, 100*np.array(skl), label=f"m$_1$/m$_2$={r/(1-r):.2f}", color=colours[i])

ax1.axhline(100*smart_optimise(m_total,avg_qber, 1e-6, np.log2(10**8) ,1.19, granularity=300),color='black', label="single block")
h=lambda qber: -qber*np.log2(qber) - (1-qber)*np.log2(1-qber)
for i,r in tqdm(enumerate(m_ratio)):
    skl=[]
    m1 = r * m_total
    m2 = m_total - m1
    for qber1 in qber1_range:
        qber2 = (avg_qber - qber1*r)/(1-r)
        if qber2<0 or qber2>0.5:
            skl.append(np.nan)
            continue
        sk_ratio1 = 1-2.19*h(qber1)
        sk_ratio2 = 1-2.19*h(qber2)
        skl.append((sk_ratio1*m1 + sk_ratio2*m2)/m_total)

    ax1.plot(qber1_range*100, 100*np.array(skl), linestyle='dashed', color=colours[i])

ax1.axhline(100*(1-2.19*h(avg_qber)),color='black', linestyle='dashed')
ax1.set_xlabel(f"QBER$_1$ (%)",fontsize=12)
ax1.set_ylabel("Secret Key Length Ratio (%)",fontsize=12)
ax1.legend()

# To specify the number of ticks on both or any single axes

ax1.xaxis.set_major_locator(plt.MaxNLocator(4))

ax1.yaxis.set_major_locator(plt.MaxNLocator(4))


f.savefig("asym_percent_double_window_comparison.pdf", dpi=300)
