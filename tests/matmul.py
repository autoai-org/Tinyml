import numpy as np

data = np.array([[0.10, 0.15, 0.20, 0.25,
                  0.30], [0.35, 0.40, 0.45, 0.55, 0.60],
                 [0.65, 0.70, 0.75, 0.80,
                  0.85], [0.90, 0.95, 0.10, 0.15, 0.20],
                 [0.25, 0.30, 0.35, 0.40, 0.45]]).reshape(1, 25)
gradient = np.array([[
    0, 0, 0, 0, 0, -0.343, -0.257, -0.0858, 0.0858, 0.2575, 0, 0.343, 0, 0, 0,
    0
]]).reshape(16, 1)

print(gradient)

output = np.matmul(gradient, data)

print(output.shape)
print(output)
