import numpy as np

theta_pretilt_lowest = 3
theta_pretilt_highest=7

print('Highest symmetry angle for green {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_highest))*(1.51+1.78)))))
print('Lowest symmetry angle for green {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_lowest))*(1.51+1.78)))))

print('Highest symmetry angle for red {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_highest))*(1.5+1.74)))))
print('Lowest symmetry angle for red {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_lowest))*(1.5+1.74)))))

print('Highest symmetry angle for blue {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_highest))*(1.53+1.86)))))
print('Lowest symmetry angle for blue {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_pretilt_lowest))*(1.53+1.86)))))





#theta_sym = 19
#print('Pretilt angle {}'.format(np.degrees(np.arcsin(np.sin(np.radians(theta_sym)/(1.78+1.52))))))