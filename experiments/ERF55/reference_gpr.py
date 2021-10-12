import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from joblib import dump
from sklearn.gaussian_process import GaussianProcessRegressor as SKGPR
from sklearn.gaussian_process.kernels import Matern

train_data = np.loadtxt("TRAIN.dat")
X, Y = train_data[:, :-1], train_data[:, -1][:, None]


kernel = Matern(nu=2.5)
model = SKGPR(kernel=kernel)
model.fit(X, Y)
dump(model, "GPR.sklearn")

# Plot Contour/Surface
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
WW = np.c_[XX.ravel(), YY.ravel()]
ZZ = model.predict(WW, return_std=False).reshape(XX.shape)
contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
cbar = fig.colorbar(contour)
plt.xlabel(r'$\xi_1$')
plt.ylabel(r'$\xi_2$')
plt.savefig("PLOTS/GPR_C.png")
plt.close()
fig_surf = go.Figure()
fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
fig_surf.add_trace(go.Scatter3d(name='Samples', x=X[:, 0], y=X[:, 1], z=Y.ravel(), mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
fig_surf.write_html("PLOTS/GPR_S.html", auto_open=False)

# Predict MCS
Xmcs = np.loadtxt("MCS.dat")[:, :-1]
MM, VV = model.predict(Xmcs, return_std=True)
print(MM.shape)
print(VV[:, None].shape)
mcs_pred = np.hstack([MM, VV[:, None]])
np.savetxt("GPR_MCS.dat", mcs_pred, delimiter='\t')
