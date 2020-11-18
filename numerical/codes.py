# %%
import numpy


# Newton method
class BasicPointSolver(object):
    """
    A basic solver for problems that input function, initial values, stop rule,
        output the result (a point result, a number), number of iterate,
        the rolling values.

    Replace those methods to generate a specific point solver:
        - input_check()
        - point_iter()
        - rule_stop()
    """

    def __init__(self, f, **kw):
        self.f = f
        self.kw = kw
        self._no_value = object()
        self.error_info = self._no_value
        self.max_iter = kw['max_iter'] if 'max_iter' in kw else 100000
        self.input_check()
        self.point_solve()

    @staticmethod
    def derivative(f, d, x):
        # two-sided difference
        # return (f(x+d)-f(x)) / d
        # one-sided difference
        return (f(x+d)-f(x-d)) / (2*d)

    @staticmethod
    def derivative2(f, d, x):
        return (f(x+2*d)-2*f(x+d)+f(x)) / d**2

    def input_check(self):
        pass

    def point_iter(self):
        pass

    def point_solve(self):
        self.n_iter = 0
        for rolling_values in self.point_iter():
            self.n_iter = self.n_iter + 1
            # TODO: save each rolling_values
            if self.rule_stop(rolling_values) or self.n_iter_stop():
                break

    def rule_stop(self, rolling_values):
        pass

    def n_iter_stop(self):
        if self.n_iter >= self.max_iter:
            self.result = numpy.NaN
            self.error_info = ('failed after iterated {} times.'
                               .format(self.n_iter))
            return True

    def __repr__(self):
        # TODO: represent more info
        if self.error_info is self._no_value:
            return 'result is {}'.format(self.result)
        else:
            return self.error_info

    def __str__(self):
        return str(self.result)


# %%
class Bisection(BasicPointSolver):
    """
    sovle x for f(x)=0 using bisection method

    f: function to be solved
    initial: a list, initial guess values
        initial[0] is initial guess xl
        initial[1] is initial guess xr
        the initial xl should be smaller than xr
    e: epsilon, the target difference between x_l and x_r
    s: sigma, the target difference between f(x) and 0
    """

    def input_check(self):
        self.e = self.kw['e'] if 'e' in self.kw else 0.001
        self.s = self.kw['s'] if 's' in self.kw else 0.00001
        # TODO: check xl < xr and f(xl)f(xr) < 0

    def point_iter(self):
        xl = self.kw['initial'][0]
        xr = self.kw['initial'][1]
        while True:
            xm = (xl+xr)/2
            [xl, xr] = [xl, xm] if self.f(xm)*self.f(xl) < 0 else [xm, xr]
            yield {'xl': xl, 'xr': xr, 'xm': xm, 'xdiff': xr-xl}

    def rule_stop(self, rolling_values):
        if (rolling_values['xdiff'] <= self.e
                and abs(self.f(rolling_values['xm'])) <= self.s):
            self.result = rolling_values['xm']
            return True


# %%
if __name__ == "__main__":
    solver = Bisection(lambda x: x**2-2*x-3, initial=[1, 10])
    solver
    print(solver)
    solver.n_iter
    solver.result


# %%
class Newton(BasicPointSolver):
    """
    sovle x for f(x)=0 using Newton's method

    f: function to be solved
    initial: a list, initial guess values
        initial[0] is initial guess x
    d: delta, the step when calculate derivatives
    e: epsilon, the target difference between x and x_last
    s: sigma, the target difference between f(x) and 0
        e should be much smaller than s
    """

    def input_check(self):
        self.d = self.kw['d'] if 'd' in self.kw else 0.0000000001
        self.e = self.kw['e'] if 'e' in self.kw else 0.00000001
        self.s = self.kw['s'] if 's' in self.kw else 0.00001
        # TODO: check self.derivative(self.f, self.d, x) != 0

    def point_iter(self):
        x = self.kw['initial'][0]
        while True:
            x_last = x
            first_order_derivative = self.derivative(self.f, self.d, x)
            if first_order_derivative == 0:
                raise ValueError('first order derivative of f at x become 0')
            else:
                x = x - self.f(x)/first_order_derivative
            yield {'x': x, 'x_last': x_last, 'xdiff': x-x_last}

    def rule_stop(self, rolling_values):
        if abs(rolling_values['xdiff']) <= self.e:
            if abs(self.f(rolling_values['x'])) <= self.s:
                self.result = rolling_values['x']
            else:
                self.result = numpy.NaN
                self.error_info = 'failed, sigma is to big.'
            return True


# %%
if __name__ == "__main__":
    solver = Newton(lambda x: x**2-2*x-3, initial=[-5])
    solver
    print(solver)
    solver.result
    solver.n_iter


# %%
class Bracketing(BasicPointSolver):
    """
    find local optimum (minimum) using using Bracketing method

    f: function to be optimized
    initial: a list, initial guess values
        initial[0] is initial guess a
        initial[1] is initial guess b
        initial[2] is initial guess c
        the initial should satisfies a < b < c, f(a), f(c)>f(b)
    e: epsilon, the target difference between x and x_last
    """

    def input_check(self):
        self.e = self.kw['e'] if 'e' in self.kw else 0.0001
        # TODO: check a < b < c, f(a), f(c)>f(b)

    def point_iter(self):
        a = self.kw['initial'][0]
        b = self.kw['initial'][1]
        c = self.kw['initial'][2]
        while True:
            d = (b+c)/2 if (b-a) < (c-b) else (a+b)/2
            if d < b:
                if self.f(d) > self.f(b):
                    a = d
                else:
                    c = b
                    b = d
            else:
                if self.f(d) < self.f(b):
                    a = b
                    b = d
                else:
                    c = d
            yield {'a': a, 'b': b, 'c': c, 'd': d, 'xdiff': c-a}

    def rule_stop(self, rolling_values):
        if rolling_values['xdiff'] <= self.e:
            self.result = rolling_values['d']
            return True


# %%
if __name__ == "__main__":
    solver = Bracketing(lambda x: x**2-2*x-3, initial=[-20, 5, 20])
    solver
    print(solver)
    solver.n_iter
    solver.result


# %%
class NewtonOptimize(Newton):
    """
    find local optimum (minimum or maxmum) using Newton's method

    f: function to be optimized
        f should satisfies first derivative and second derivative both exist
        and are both continuous
    initial: a list, initial guess values
        initial[0] is initial guess xl
        initial[1] is initial guess xr
        the initial xl should be smaller than xr
    d: delta, the step when calculate derivatives
    e: epsilon, the target difference between x and x_last
    s: sigma, the target difference between f(x) and 0
        e should be much smaller than s
    """

    def input_check(self):
        self.d = self.kw['d'] if 'd' in self.kw else 0.0000000001
        self.e = self.kw['e'] if 'e' in self.kw else 0.00000001
        self.s = self.kw['s'] if 's' in self.kw else 0.00001
        # TODO: check self.derivative2(self.f, self.d, x) != 0
        # TODO: check f:
        # f should satisfies first derivative and second derivative both exist
        # and are both continuous

    def point_iter(self):
        x = self.kw['initial'][0]
        while True:
            x_last = x
            first_order_derivative = self.derivative(self.f, self.d, x)
            second_order_derivative = self.derivative2(self.f, self.d, x)
            if second_order_derivative == 0:
                raise ValueError('second order derivative of f at x become 0')
            else:
                x = x - first_order_derivative/second_order_derivative
            yield {'x': x, 'x_last': x_last, 'xdiff': x-x_last}

    def rule_stop(self, rolling_values):
        if abs(rolling_values['xdiff']) <= self.e:
            if (abs(self.derivative(self.f, self.d, rolling_values['x'])) <=
                    self.s):
                self.result = rolling_values['x']
            else:
                self.result = numpy.NaN
                self.error_info = 'failed, sigma is to big.'
            return True
