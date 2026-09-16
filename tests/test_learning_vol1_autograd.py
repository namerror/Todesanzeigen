import math
import sys
from pathlib import Path
from unittest import TestCase


LEARNING_VOL1 = Path(__file__).resolve().parents[1] / "learning" / "vol1"
sys.path.insert(0, str(LEARNING_VOL1))

from local_derivatives import central_difference  # noqa: E402
from scalar_value_graph import Value  # noqa: E402


def _chapter_three_graph(a_data: float, b_data: float, c_data: float):
    a = Value(a_data, label="a")
    b = Value(b_data, label="b")
    c = Value(c_data, label="c")

    loss = (a * b + c).tanh() + a**2 + (1 + b.exp()).log()
    return loss, {"a": a, "b": b, "c": c}


def _chapter_three_plain(a: float, b: float, c: float) -> float:
    return math.tanh(a * b + c) + a**2 + math.log(1 + math.exp(b))


class LearningVol1AutogradTests(TestCase):
    def assertCloseToCentralDifference(
        self,
        analytic: float,
        f,
        x: float,
        *,
        places: int = 5,
    ) -> None:
        numeric = central_difference(f, x)
        self.assertAlmostEqual(analytic, numeric, places=places)

    def test_mixed_value_and_number_arithmetic_matches_python(self) -> None:
        x = Value(2.0)

        self.assertEqual((x + 3).data, 5.0)
        self.assertEqual((3 + x).data, 5.0)
        self.assertEqual((x * 3).data, 6.0)
        self.assertEqual((3 * x).data, 6.0)
        self.assertEqual((x - 3).data, -1.0)
        self.assertEqual((3 - x).data, 1.0)
        self.assertEqual((x / 4).data, 0.5)
        self.assertEqual((4 / x).data, 2.0)

    def test_branching_addition_accumulates_gradient(self) -> None:
        x = Value(4.0)

        y = x + x
        y.backward()

        self.assertEqual(y.data, 8.0)
        self.assertEqual(x.grad, 2.0)

    def test_branching_multiplication_matches_central_difference(self) -> None:
        x = Value(3.0)

        y = x * x
        y.backward()

        self.assertEqual(y.data, 9.0)
        self.assertCloseToCentralDifference(x.grad, lambda value: value * value, 3.0)

    def test_reused_node_preserves_both_paths(self) -> None:
        a = Value(2.0)
        b = Value(-3.0)

        d = a * b + a
        d.backward()

        self.assertEqual(d.data, -4.0)
        self.assertEqual(a.grad, -2.0)
        self.assertEqual(b.grad, 2.0)

    def test_unary_operations_match_central_difference_at_safe_points(self) -> None:
        x = Value(0.7)
        y = x.tanh() + x.exp() + x.log()

        y.backward()

        def plain(value: float) -> float:
            return math.tanh(value) + math.exp(value) + math.log(value)

        self.assertCloseToCentralDifference(x.grad, plain, 0.7)

    def test_division_and_power_match_central_difference(self) -> None:
        x = Value(1.5)
        y = (x**3) / 2 + 6 / x

        y.backward()

        def plain(value: float) -> float:
            return (value**3) / 2 + 6 / value

        self.assertCloseToCentralDifference(x.grad, plain, 1.5)

    def test_chapter_three_graph_leaf_gradients_match_central_difference(self) -> None:
        values = {"a": 2.0, "b": 0.0, "c": -1.0}
        loss, leaves = _chapter_three_graph(**{f"{name}_data": value for name, value in values.items()})

        loss.backward()

        for name, leaf in leaves.items():
            def perturbed(value: float, name=name) -> float:
                args = dict(values)
                args[name] = value
                return _chapter_three_plain(args["a"], args["b"], args["c"])

            self.assertCloseToCentralDifference(leaf.grad, perturbed, values[name])

