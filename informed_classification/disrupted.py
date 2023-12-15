import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, multivariate_normal

# Set parameters for chi-square and multivariate Gaussian
df_chi2 = 3  # degrees of freedom for chi-square
mean_gaussian = [1, 2]  # mean for multivariate Gaussian
covariance_matrix = [[1, 0.5], [0.5, 2]]  # covariance matrix for multivariate Gaussian

# Define the range for the plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the joint PDF
chi_square_pdf = chi2.pdf(X, df_chi2)
gaussian_pdf = multivariate_normal.pdf(pos, mean=mean_gaussian, cov=covariance_matrix)
joint_pdf = chi_square_pdf * gaussian_pdf

# Normalize the joint PDF
normalization_constant = np.trapz(np.trapz(joint_pdf, x), y)
normalized_joint_pdf = joint_pdf / normalization_constant

# Create a filled contour plot
plt.contourf(X, Y, normalized_joint_pdf, levels=20, cmap='viridis')
plt.title('Custom Distribution Filled Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Probability Density')
plt.show()
     