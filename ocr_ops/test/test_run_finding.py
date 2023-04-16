import unittest

from ocr_ops.run_finding.interval import Interval
from ocr_ops.run_finding.run_finding import (
    find_uninterrupted_runs,
    stitch_with_tol,
    find_runs_with_tol,
)


class TestRunFinding(unittest.TestCase):
    def test_interval(self) -> None:
        """
        Test interval class.
        """

        # test interval properties
        interval = Interval(0, 1)
        self.assertEqual(interval.start, 0)
        self.assertEqual(interval.end, 1)
        self.assertEqual(interval.__str__(), "0-1")
        self.assertEqual(interval.__eq__(Interval(0, 1)), True)
        self.assertEqual(interval.__eq__(Interval(1, 1)), False)
        self.assertEqual(interval.__eq__(Interval(0, 2)), False)
        self.assertEqual(interval.__eq__(Interval(1, 2)), False)
        self.assertEqual(len(interval), 1)

        # test two adjacent intervals
        interval_1 = Interval(0, 1)
        interval_2 = Interval(1, 2)
        self.assertEqual(interval_1.is_disjoint(interval_2), True)
        self.assertEqual(interval_2.is_disjoint(interval_1), True)

        # test interval equals
        interval_1 = Interval(0, 1)
        interval_2 = Interval(0, 1)
        self.assertEqual(interval_1.equals(interval_2), True)

        # test interval contains
        interval_1 = Interval(0, 1)
        self.assertEqual(interval_1.contains(0), True)
        self.assertEqual(interval_1.contains(1), False)
        self.assertEqual(interval_1.contains(2), False)

        # test interval intersects
        interval_1 = Interval(0, 1)
        interval_2 = Interval(1, 2)
        self.assertEqual(interval_1.intersects(interval_2), False)
        self.assertEqual(interval_2.intersects(interval_1), False)

        # test interval area overlap
        interval_1 = Interval(0, 1)
        interval_2 = Interval(1, 2)
        with self.assertRaises(ValueError):
            interval_1.overlap_interval(interval_2)
        with self.assertRaises(ValueError):
            interval_2.overlap_interval(interval_1)
        interval_1 = Interval(0, 2)
        interval_2 = Interval(1, 3)
        self.assertEqual(interval_1.overlap_interval(interval_2), Interval(1, 2))

        # test interval percent overlap
        interval_1 = Interval(0, 2)
        interval_2 = Interval(1, 3)
        self.assertEqual(interval_1.percent_overlap(interval_2), 0.5)

    def test_uninterrupted_runs(self) -> None:
        """
        Test uninterrupted runs.
        """
        # test basic
        series = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 3]
        runs_1 = find_uninterrupted_runs(series=series, query_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, query_elem=2)
        runs_3 = find_uninterrupted_runs(series=series, query_elem=3)
        self.assertListEqual(runs_1, [Interval(0, 3), Interval(6, 10)])
        self.assertListEqual(runs_2, [Interval(3, 6)])
        self.assertListEqual(runs_3, [Interval(10, 11)])

        # test no run
        series = [1, 1, 1]
        runs_1 = find_uninterrupted_runs(series=series, query_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, query_elem=2)
        self.assertListEqual(runs_1, [Interval(0, 3)])
        self.assertListEqual(runs_2, [])

        # test empty series
        series = []
        runs_0 = find_uninterrupted_runs(series=series, query_elem=0)
        self.assertListEqual(runs_0, [])

        # test single elem
        series = [1]
        runs_1 = find_uninterrupted_runs(series=series, query_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, query_elem=2)
        self.assertListEqual(runs_1, [Interval(0, 1)])
        self.assertListEqual(runs_2, [])

        # test 2 elem
        series = [1, 2]
        runs_1 = find_uninterrupted_runs(series=series, query_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, query_elem=2)
        self.assertListEqual(runs_1, [Interval(0, 1)])
        self.assertListEqual(runs_2, [Interval(1, 2)])

        # test 3 elem
        series = [1, 2, 3]
        runs_1 = find_uninterrupted_runs(series=series, query_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, query_elem=2)
        runs_3 = find_uninterrupted_runs(series=series, query_elem=3)
        self.assertListEqual(runs_1, [Interval(0, 1)])
        self.assertListEqual(runs_2, [Interval(1, 2)])
        self.assertListEqual(runs_3, [Interval(2, 3)])

    def test_stitcher(self) -> None:
        """
        Test stitcher.
        """
        # basic test
        runs = [
            Interval(0, 1),
            Interval(2, 5),
            Interval(10, 12),
            Interval(13, 14),
            Interval(15, 16),
            Interval(100, 101),
        ]
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=4)
        self.assertListEqual(
            stitched, [Interval(0, 5), Interval(10, 16), Interval(100, 101)]
        )
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=5)
        self.assertListEqual(stitched, [Interval(0, 16), Interval(100, 101)])
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=100)
        self.assertListEqual(stitched, [Interval(0, 101)])
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=0)
        self.assertListEqual(stitched, runs)

        # empty runs
        runs = []
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=0)
        self.assertListEqual(stitched, [])

        # test err
        with self.assertRaises(ValueError):
            stitch_with_tol(runs, tol=-1)

    def test_find_runs_with_tol(self) -> None:
        """
        Test find runs with tolerance.
        """
        # test basic
        series = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 3]
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=0),
            [Interval(0, 3), Interval(6, 10)],
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=2),
            [Interval(0, 3), Interval(6, 10)],
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=3), [Interval(0, 10)]
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=4), [Interval(0, 10)]
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=2, tol=4), [Interval(3, 6)]
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=3, tol=4), [Interval(10, 11)]
        )
        self.assertListEqual(find_runs_with_tol(series=series, query_elem=4, tol=4), [])

        # test 1 elem
        series = [1]
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=1),
            [Interval(0, 1)],
        )
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=2, tol=1),
            [],
        )

        # test empty
        series = []
        self.assertListEqual(
            find_runs_with_tol(series=series, query_elem=1, tol=1),
            [],
        )
