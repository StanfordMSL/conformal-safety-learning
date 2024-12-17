import BasicTools.geometric_helpers as geom
import numpy as np
from Transformers.PCAtransformer import PCATransformer
from Transformers.KPCAtransformer import KPCATransformer
import matplotlib.pyplot as plt
import BasicTools.helpers as hp
from sklearn.datasets import make_swiss_roll

def generate_sphere(num_draws, dim, center=0, r=1, noise_scale=0):
    samples = center + r * geom.gen_unit_vectors(num_draws, dim)
    samples += np.random.normal(loc=0, scale=noise_scale, size=samples.shape)
    return samples

np.random.seed(0)

# Generate training data on circle points in 2D
dim = 3
sphere1 = generate_sphere(num_draws=500, dim=dim, center=0, r=1, noise_scale=0.0)
# sphere2 = generate_sphere(num_draws=500, dim=dim, center=np.array([0.8]*dim), noise_scale=0.0)
sphere2 = generate_sphere(num_draws=500, dim=dim, center=0, r=3, noise_scale=0.0)

# Different data options:
# 1. A single shere
# data = sphere1
# 2. Two spheres
# data = np.vstack([sphere1, sphere2])
# 3. Swiss roll dataset
data = make_swiss_roll(n_samples=500, noise=0.0, hole=False)[0]

# Make a fake trajectory so that can transform
traj = hp.Trajectory(data, data, 'success', data)
rollouts = hp.Rollouts([traj])

# Different transformation options:
# 1. Fit KPCA to training data
transformer = KPCATransformer(n_components=5, kernel='rbf', gamma='auto', weight=False)
# 2. Fit PCA to training data
# transformer = PCATransformer(n_components=dim, weight=False)
transformer.fit(rollouts)

# print('transformer.gamma', transformer.gamma)

# Plot the eigenvalues and the max eigengap
fig, ax = plt.subplots()

ax.plot(range(1,transformer.n_components+1),transformer.D)
ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Eigenvalues')

eigengaps = transformer.D[:-1] - transformer.D[1:]
max_gap_ind = np.argmax(eigengaps)
ax.vlines(x=max_gap_ind+1, ymin=0, ymax=transformer.D[0], linestyle='dashed', label='Max Eigengap')

# Plot each feature separately as heatmap in the original space
features = transformer.apply(data)

fig = plt.figure()

for k in range(transformer.n_components):
    if dim == 2:
        ax = fig.add_subplot(1,transformer.n_components, k+1)
    elif dim == 3:
        ax = fig.add_subplot(1,transformer.n_components, k+1, projection='3d')

    ax.scatter(*data.T.tolist(), c=features[:,k])
    ax.set_title(f'Feature {k}: Eigenvalue = {np.round(transformer.D[k],5)}')

plt.show()
