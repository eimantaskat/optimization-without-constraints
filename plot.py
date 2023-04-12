import numpy as np
import matplotlib.pyplot as plt

def create_plot(f, points=None, x_min=-1, x_max=1, y_min=-1, y_max=1):
	# Create a grid of points
	X = np.linspace(x_min, x_max, 100)
	Y = np.linspace(y_min, y_max, 100)
	X, Y = np.meshgrid(X, Y)
	Z = f([X, Y])

	levels = [-0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2]

	# Create a contour plot
	cp = plt.contour(X, Y, Z, colors='black', levels=levels)

	if points is not None:
		for x_point, y_point in points:
			plt.scatter(x_point, y_point)

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Contour map')
	plt.show()

def add_point(ax, x, y, z, color='black', marker='o', size=10, zorder=10):
	ax.scatter(x, y, z, color=color, marker=marker, s=size, zorder=zorder)

def create_3d_plot(f, points=None, x_min=-1, x_max=1, y_min=-1, y_max=1):
	# Create a grid of points
	X = np.linspace(x_min, x_max, 100)
	Y = np.linspace(y_min, y_max, 100)
	X, Y = np.meshgrid(X, Y)
	Z = f([X, Y])

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z)

	if points is not None:
		for x_point, y_point in points:
			z_point = f([x_point, y_point])
			add_point(ax, x_point, y_point, z_point)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show(block=True)
