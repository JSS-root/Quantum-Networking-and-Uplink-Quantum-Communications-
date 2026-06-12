import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
f, ax1  = plt.subplots(1, 1, figsize=(5, 3))
phi_data,_,skl = np.loadtxt("data/processed/opt_skl_15deg_500000m_0m_0.3m_0km.csv",skiprows=1,delimiter=",").T
ax2=ax1.twiny()
R,h=6371, 550
theta = lambda d,R,h: np.arctan((np.cos(d/R) - R/(R + h))/np.sin(d/R))
d_func = lambda phi,R,h: R*(-phi + np.arccos((R * np.cos(phi))/(h+R)))
d= np.linspace(d_func(phi_data[0]*np.pi/180, R, h), d_func(phi_data[-1]*np.pi/180, R, h), len(phi_data))
obj = lambda x, a, b, c, d, e,f,g: g*x**5+f*x**4 +a*x**3+b*x**2+c*x+d*x+e
popt,cov=curve_fit(obj, phi_data, skl)


popt,cov=curve_fit(obj, phi_data, skl,p0=popt)
ax1.plot(d,obj(180/np.pi*theta(d, R, h),*popt),color="red",label="0 km - Micius")
ax1.fill_between(d, d*0, y2=obj(180/np.pi*theta(d, R, h),*popt),hatch="/",color="red",alpha=0.15)

phis= np.arange(phi_data[-1], phi_data[0]-1,-15)[::-1]*np.pi/180
ax2.set_xticks([*d_func(phis, R, h),d_func(15*np.pi/180, R, h)])
ax2.set_xticklabels([rf"{di:.0f}" for di in phis*180/np.pi]+[15])
#####
obj = lambda x, a, b, c, d, e,f,g,h,i: i*x**7+h*x**6+g*x**5+f*x**4 +a*x**3+b*x**2+c*x+d*x+e

phi_data,_,skl = np.loadtxt("data/processed/opt_skl_15deg_500000m_0m_0.3m_10km.csv",skiprows=1,delimiter=",").T
d= np.linspace(d_func(phi_data[0]*np.pi/180, R, h), d_func(phi_data[-1]*np.pi/180, R, h), len(phi_data))
popt,cov=curve_fit(obj, phi_data, skl)
print(popt)
ax1.plot(d,obj(180/np.pi*theta(d, R, h),*popt),"--",color="red",label="10 km - Micius")
ax1.fill_between(d, d*0, y2=obj(180/np.pi*theta(d, R, h),*popt),hatch="/",color="red",alpha=0.15)
#####
#####
phi_data,_,skl = np.loadtxt("data/processed/opt_skl_15deg_550000m_0m_0.25m_0km.csv",skiprows=1,delimiter=",").T
d= np.linspace(d_func(phi_data[0]*np.pi/180, R, h), d_func(phi_data[-1]*np.pi/180, R, h), len(phi_data))
popt,cov=curve_fit(obj, phi_data, skl)

ax1.plot(d,obj(180/np.pi*theta(d, R, h),*popt),color="blue",label="0 km - QEYSSat")
ax1.fill_between(d, d*0, y2=obj(180/np.pi*theta(d, R, h),*popt),hatch='\\',color="blue",alpha=0.15)
#####
#####
phi_data,_,skl = np.loadtxt("data/processed/opt_skl_15deg_550000m_0m_0.25m_10km.csv",skiprows=1,delimiter=",").T
d= np.linspace(d_func(phi_data[0]*np.pi/180, R, h), d_func(phi_data[-1]*np.pi/180, R, h), len(phi_data))
popt,cov=curve_fit(obj, phi_data, skl)

ax1.plot(d,obj(180/np.pi*theta(d, R, h),*popt),"--",color="blue",label="10 km - QEYSSat")
ax1.fill_between(d, d*0, y2=obj(180/np.pi*theta(d, R, h),*popt),hatch='\\',color="blue",alpha=0.15)
#####

ax1.legend()
ax1.set_ylabel(r"SKL (bits)")
ax1.set_xlabel(r"d (km)")
ax1.set_xlim(0, d_func(15*np.pi/180, R, h))
# ax1.set_ylim(0,1.1*max(skl))
ax1.set_yscale("log")
ax2.set_xlabel(r"$\phi_{\text{max}} (^\circ$)")
f.savefig("gto_wofibre_Micius_vs_QEYSSat.pdf",dpi=300,bbox_inches="tight")


# f, ax1  = plt.subplots(1, 1, figsize=(3, 2))
# phi,paa = np.loadtxt("PAA.csv",skiprows=3,delimiter=",").T
# ax2=ax1.twiny()

# d= np.linspace(0, d_func(0, R, h), 100)
# ax1.plot(d,theta(d, R, h),color="blue")
# ax1.set_ylabel(r"SKL (cps)")
# ax1.set_xlabel(r"d (m)")
# phis= np.arange(0, 91, 15)*np.pi/180
# ax2.set_xticks(d_func(phis, R, h))
# ax2.set_xticklabels([rf"{di:i}$^\circ$" for di in phis*180/np.pi])
# f.savefig("test_90.pdf",dpi=300,bbox_inches="tight")
