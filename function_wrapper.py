import numpy as np
import sympy as sp
from sympy.core.numbers import Rational


class Function(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		function = kwargs.get('f', None)
		symbols = kwargs.get('symbols', None)
		if symbols:
			self._x = symbols
		else:
			self._x = sp.symbols('x')
		self._function = sp.sympify(function)
		self.clear()

	def __missing__(self, key):
		if isinstance(key, tuple):
			subs_list = []
			for i in range(len(key)):
				subs_list.append((self._x[i], key[i]))
			value = self._function.subs(subs_list)
		else:
			value = self._function.subs(self._x, key)
		self[key] = value
		return float(value)

	@property
	def times_called(self):
		keys = self.keys()
		return len(keys)

	@property
	def f(self):
		return self._function


class FunctionWrapper():
	def __init__(self, *args, **kwargs):
		function = kwargs.get('function', None)
		self._derivatives = {}
		symbols = kwargs.get('gradient_symbols', None)

		if symbols:
			self._function = Function(f=function, symbols=symbols)
			self._symbols = sp.symbols(symbols)

			self._gradient = {}
			for symbol in self._symbols:
				self._gradient[symbol] = Function(f=sp.diff(self._function.f, symbol), symbols=self._symbols)
		else:
			self._function = Function(f=function, symbols=symbols)

		self._gradient_points = []


	@property
	def function(self):
		return self._function
	
	@property
	def derivatives(self):
		return self._derivatives
	
	@property
	def gradient(self):
		return self._gradient
	
	@property
	def gradient_points(self):
		return self._gradient_points

	def at(self, x: float | list[float]):
		"""
		Computes the value of the function at a point x.
		"""
		if not self._function.f:
			raise ValueError("No function defined")
		if isinstance(x, (int, float, Rational)):
			return self._function[x]
		else:
			if isinstance(x, np.ndarray):
				return self._function[*x]
			return [self._function[i] for i in x]

	def dx_at(self, x: float | list[float], order: int = 1):
		"""
		Computes the value of the n-th derivative of the function at a point x.
		"""
		if not self._derivatives.get(order):
			self._derivatives[order] = Function(
				f=sp.diff(self._function.f, sp.symbols('x'), order))
		if isinstance(x, (int, float, Rational)):
			return self._derivatives[order][x]
		else:
			return [self._derivatives[order][i] for i in x]
		
	def gradient_at(self, x: float | list[float]):
		"""
		Computes the value of the gradient of the function at a point x.
		"""
		if not self._gradient:
			raise ValueError("No gradient defined")
		if len(x) != len(self._symbols):
			raise ValueError(f'Incorrect number of arguments: Expected {len(self._symbols)} but got {len(x)})')
		
		self._gradient_points.append(x)

		result = []
		for symbol in self._symbols:
			y = self._gradient[symbol][*x]
			result.append(float(y))
		return np.array(result)
	
	def clear_cache(self):
		self._function.clear()
		for derivative in self._derivatives.values():
			derivative.clear()
		for gradient in self._gradient.values():
			gradient.clear()
		self._gradient_points = []
