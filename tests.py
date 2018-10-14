from copy import deepcopy

import unittest
import numpy as np

from .mytracing import equal_pt_indices, equal_points, Bifurcation

TOL = 1e-9


class EqualPtIndices(unittest.TestCase):

    def test1(self):
        points1 = [(5,5), (4,4), (3,3), (2,2), (1,1)]
        points2 = [(5,5), (4,4), (3,3), (2,4), (1,5)]

        res1, res2 = equal_pt_indices(points1, points2, reverse=False)
        self.assertEqual(res1, 0)
        self.assertEqual(res2, 0)

    def test1_reverse(self):
        points1 = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
        points2 = [(5, 5), (4, 4), (3, 3), (2, 4), (1, 5)]

        res1, res2 = equal_pt_indices(points1, points2, reverse=True)
        self.assertEqual(res1, 2)
        self.assertEqual(res2, 2)


class EqualPoints(unittest.TestCase):

    def test_false(self):
        points1 = [(5,5), (4,4), (3,3), (2,2), (1,1)]
        points2 = [(5,5), (4,4), (3,3), (2,4), (1,5)]

        res1 = equal_points(points1, points2)
        self.assertFalse(res1)

    def test_true(self):
        points1 = [(5,5), (4,4), (3,3), (2,2), (1,1)]
        points2 = [(5,5), (4,4), (3,3), (2,2), (1,1)]

        res1 = equal_points(points1, points2)
        self.assertTrue(res1)


class BifurcationMergeTest(unittest.TestCase):

    def test_simple(self):
        points1 = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
        points2 = [(5, 5), (4, 4), (3, 3), (2, 4), (1, 5)]

        bif1 = Bifurcation(points1)
        bif2 = Bifurcation(points2)

        res_success, res_merged = Bifurcation.merge(bif1=bif1, bif2=bif2)
        self.assertTrue(res_success)

        expected_points = [(5, 5), (4, 4), (3, 3)]
        expected_left = [(2, 2), (1, 1)]
        expected_right = [(2, 4), (1, 5)]

        points_same = np.array_equal(res_merged.points, expected_points)
        if not points_same:
            print('points: ', res_merged.points)

        left_same = np.array_equal(res_merged.children[0].points, expected_left)
        if not left_same:
            print('left: ', res_merged.children[0])

        right_same = np.array_equal(res_merged.children[1].points, expected_right)
        if not right_same:
            print('right: ', res_merged.children[1])

        self.assertEqual(len(res_merged.children), 2)
        self.assertEqual(len(res_merged.children[0].children), 0)
        self.assertEqual(len(res_merged.children[1].children), 0)

        self.assertTrue(points_same)
        self.assertTrue(left_same)
        self.assertTrue(right_same)

    def test_merge_to_right_child(self):
        points1 = [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
        points2 = [(7, 7), (6, 6), (5, 5), (4, 6), (3, 6), (2, 6), (1, 6)]
        points3 = [(7, 7), (6, 6), (5, 5), (4, 6), (3, 6), (2, 7), (1, 7)]

        bif1 = Bifurcation(points1)
        bif2 = Bifurcation(points2)
        bif3 = Bifurcation(points3)

        res_success, res_merged = Bifurcation.merge(bif1=bif1, bif2=bif2)
        self.assertTrue(res_success)

        expected_points = [(7, 7), (6, 6), (5, 5)]
        expected_left = [(4, 4), (3, 3), (2, 2), (1, 1)]
        expected_right = [(4, 6), (3, 6), (2, 6), (1, 6)]

        points_same = np.array_equal(res_merged.points, expected_points)
        left_same = np.array_equal(res_merged.children[0].points, expected_left)
        right_same = np.array_equal(res_merged.children[1].points, expected_right)

        self.assertEqual(len(res_merged.children), 2)
        self.assertEqual(len(res_merged.children[0].children), 0)
        self.assertEqual(len(res_merged.children[1].children), 0)

        self.assertTrue(points_same)
        self.assertTrue(left_same)
        self.assertTrue(right_same)

        res_success2, res_merged2 = Bifurcation.merge(bif1=res_merged, bif2=bif3)
        self.assertTrue(res_success2)

        expected_points = [(7, 7), (6, 6), (5, 5)]
        expected_left = [(4, 4), (3, 3), (2, 2), (1, 1)]
        expected_right_points = [(4, 6), (3, 6)]
        expected_right_left = [(2, 6), (1, 6)]
        expected_right_right = [(2, 7), (1, 7)]

        points_same = np.array_equal(res_merged2.points, expected_points)
        self.assertTrue(points_same)
        self.assertEqual(len(res_merged2.children), 2)
        self.assertEqual(len(res_merged2.children[0].children), 0)
        self.assertEqual(len(res_merged2.children[1].children), 2)

        left_same = np.array_equal(res_merged2.children[0].points, expected_left)
        self.assertTrue(left_same)
        self.assertEqual(len(res_merged2.children), 2)

        right_points_same = np.array_equal(res_merged2.children[1].points, expected_right_points)
        self.assertTrue(right_points_same)
        self.assertEqual(len(res_merged2.children[1].children), 2)

        right_left_same = np.array_equal(res_merged2.children[1].children[0].points, expected_right_left)
        self.assertTrue(right_left_same)

        right_right_same = np.array_equal(res_merged2.children[1].children[1].points, expected_right_right)
        self.assertTrue(right_right_same)

        self.assertEqual(len(res_merged2.children[1].children[0].children), 0)
        self.assertEqual(len(res_merged2.children[1].children[1].children), 0)

    def test_merge_to_points(self):
        points1 = [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
        points2 = [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 4), (1, 5)]
        points3 = [(7, 7), (6, 6), (5, 5), (4, 6), (3, 6), (2, 7), (1, 7)]

        bif1 = Bifurcation(points1)
        bif2 = Bifurcation(points2)
        bif3 = Bifurcation(points3)

        res_success, res_merged = Bifurcation.merge(bif1=bif1, bif2=bif2)
        self.assertTrue(res_success)

        expected_points = [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3)]
        expected_left = [(2, 2), (1, 1)]
        expected_right = [(2, 4), (1, 5)]

        points_same = np.array_equal(res_merged.points, expected_points)
        left_same = np.array_equal(res_merged.children[0].points, expected_left)
        right_same = np.array_equal(res_merged.children[1].points, expected_right)

        self.assertEqual(len(res_merged.children), 2)
        self.assertEqual(len(res_merged.children[0].children), 0)
        self.assertEqual(len(res_merged.children[1].children), 0)
        self.assertTrue(points_same)
        self.assertTrue(left_same)
        self.assertTrue(right_same)

        res_success2, res_merged2 = Bifurcation.merge(bif1=res_merged, bif2=bif3)
        self.assertTrue(res_success2)

        expected_points = [(7, 7), (6, 6), (5, 5)]
        expected_left = [(4, 4), (3, 3)]
        expected_left_left = [(2, 2), (1, 1)]
        expected_left_right = [(2, 4), (1, 5)]

        expected_right = [(4, 6), (3, 6), (2, 7), (1, 7)]

        points_same = np.array_equal(res_merged2.points, expected_points)
        self.assertTrue(points_same)
        self.assertEqual(len(res_merged2.children), 2)
        self.assertEqual(len(res_merged2.children[0].children), 2)
        self.assertEqual(len(res_merged2.children[0].children[0].children), 0)
        self.assertEqual(len(res_merged2.children[0].children[1].children), 0)
        self.assertEqual(len(res_merged2.children[1].children), 0)

        left_same = np.array_equal(res_merged2.children[0].points, expected_left)
        self.assertTrue(left_same)

        right_points_same = np.array_equal(res_merged2.children[1].points, expected_right)
        self.assertTrue(right_points_same)

        left_left_same = np.array_equal(res_merged2.children[0].children[0].points, expected_left_left)
        self.assertTrue(left_left_same)

        left_right_same = np.array_equal(res_merged2.children[0].children[1].points, expected_left_right)
        self.assertTrue(left_right_same)

        # check that get_points() doesnt change the points filed anymore
        orig_points = deepcopy(res_merged2.points)
        res_merged2.get_points()
        new_points = res_merged2.points
        self.assertTrue(np.array_equal(orig_points, new_points))

        self.assertEqual(len(res_merged2.get_points()), 13)


if __name__ == '__main__':
    unittest.main()
