import numpy as np

from optimization_algorithms import gradient_descent, steepest_descent, simplex
from function_wrapper import FunctionWrapper
from plot import create_plot, create_3d_plot

USE_3D_PLOT = False


def f(X):
	return -1 * ((1 - X[0] - X[1]) * X[0] * X[1]) / 8


if __name__ == '__main__':
	func = '-1 * ((1 - x - y) * x * y) / 8'
	fw = FunctionWrapper(function=func, gradient_symbols=['x', 'y'])

	a = 5
	b = 0

	alpha = .3

	if USE_3D_PLOT:
		plot = create_3d_plot
	else:
		plot = create_plot

	# plot = lambda f, **kwargs: None

	print('Gradient descent:')
	x = np.array([0, 0])
	result, iter_count = gradient_descent(fw, x, 0.9, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='gradient_descent_0_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([1, 1])
	result, iter_count = gradient_descent(fw, x, 0.9, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='gradient_descent_1_1')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([a/10, b/10])
	result, iter_count = gradient_descent(fw, x, 0.9, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='gradient_descent_05_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	print('Steepest descent:')
	x = np.array([0, 0])
	result, iter_count = steepest_descent(fw, x, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='steepest_descent_0_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([1, 1])
	result, iter_count = steepest_descent(fw, x, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='steepest_descent_1_1')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([a/10, b/10])
	result, iter_count = steepest_descent(fw, x, 1e-6)
	points = fw.gradient_points# + list(fw.function.keys())
	plot(f, points=points, name='steepest_descent_05_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	print('Simplex method:')
	x = np.array([0, 0])
	result, iter_count = simplex(fw, x, alpha, 1e-6)
	points = fw.gradient_points + list(fw.function.keys())
	plot(f, points=points, name='simplex_0_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([1, 1])
	result, iter_count = simplex(fw, x, alpha, 1e-6)
	points = fw.gradient_points + list(fw.function.keys())
	plot(f, points=points, name='simplex_1_1')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()

	x = np.array([a/10, b/10])
	result, iter_count = simplex(fw, x, alpha, 1e-6)
	points = fw.gradient_points + list(fw.function.keys())
	plot(f, points=points, name='simplex_05_0')
	times_called = fw.function.times_called
	print(f'{x}:', result)
	print(f'f({result}): {fw.at(result)}')
	print(f'Iterations: {iter_count}')
	print(f'f(x) calculated {times_called} times')
	print(f'Gradient calculated {fw.gradient[next(iter(fw.gradient))].times_called} times\n')
	fw.clear_cache()
