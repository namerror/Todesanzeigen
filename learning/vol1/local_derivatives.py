import math

# available functions
def f(x):
    return x**2

def g(x):
    return x

def h(x):
    return 3 * x + 2

def sin(x):
    return math.sin(x)

def tanh(x):
    return math.tanh(x)

def log(x):
    return math.log(x)

def central_difference(f, x: float, h: float = 1e-6) -> float:
    """Return a centered finite-difference estimate of df/dx."""
    return (f(x + h) - f(x - h)) / (2 * h)

if __name__ == "__main__":
    # verify that the central difference approximation is working correctly
    x = 2.0
    print(f"x = {x}")
    print("Central difference approximations:")

    print(f"f'({x}) = {central_difference(f, x)} (expected: {2 * x})")
    print(f"g'({x}) = {central_difference(g, x)} (expected: {1})")
    print(f"h'({x}) = {central_difference(h, x)} (expected: {3})")
    print(f"sin'({x}) = {central_difference(sin, x)} (expected: {math.cos(x)})")
    print(f"tanh'({x}) = {central_difference(tanh, x)} (expected: {1 - math.tanh(x)**2})")
    print(f"log'({x}) = {central_difference(log, x)} (expected: {1 / x})")

    # check if central difference is better than forward difference
    def forward_difference(f, x: float, h: float = 1e-6) -> float:
        """Return a forward finite-difference estimate of df/dx."""
        return (f(x + h) - f(x)) / h

    print("\nForward difference approximations:")

    print(f"f'({x}) = {forward_difference(f, x)} (expected: {2 * x})")
    print(f"g'({x}) = {forward_difference(g, x)} (expected: {1})")
    print(f"h'({x}) = {forward_difference(h, x)} (expected: {3})")
    print(f"sin'({x}) = {forward_difference(sin, x)} (expected: {math.cos(x)})")
    print(f"tanh'({x}) = {forward_difference(tanh, x)} (expected: {1 - math.tanh(x)**2})")
    print(f"log'({x}) = {forward_difference(log, x)} (expected: {1 / x})")

    # check how different h values affect the accuracy of the central difference approximation
    
    print(f"\nCentral difference approximations with different h values (tanh):")
    errors = []
    h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for h in h_values:
        errors.append(abs(central_difference(tanh, x, h) - (1 - math.tanh(x)**2)))
        print(f"h = {h}: f'({x}) = {central_difference(tanh, x, h)} (expected: {1 - math.tanh(x)**2})")

    print("\nBest h value for central difference approximation (tanh):")
    print(f"h = {h_values[errors.index(min(errors))]} with error = {min(errors)}")