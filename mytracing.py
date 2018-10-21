from operator import itemgetter
import copy
from collections import OrderedDict
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt


def same_point(p1, p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return True
    else:
        return False


def equal_points(l1, l2):
    if len(l1) != len(l2):
        return False
    for e1, e2 in zip(l1, l2):
        if not same_point(e1, e2):
            return False
    return True


def equal_pt_indices(l1, l2, reverse=False):
    """
    It returns the indices for each list of the first (according to list 1) common element. It is assumed that
    elements are unique within the lists.
    :param l1: first list
    :param l2: second list
    :param reverse: if True, it is the last instead of the first
    :return: it returns the pair of indices for 2 lists. If not found, it returns (-1, -1)
    """
    ind_dict1 = dict((e, i) for i, e in enumerate(l1))
    ind_dict2 = dict((e, i) for i, e in enumerate(l2))

    intersected_elements = set(ind_dict1).intersection(set(ind_dict2))
    indices = [(ind_dict1[e], ind_dict2[e]) for e in intersected_elements]
    indices.sort(key=itemgetter(0), reverse=reverse)

    if len(indices) < 1:
        return -1, -1
    else:
        return indices[0]


class Bifurcation:

    def __init__(self, points):
        """
        Bifurcation object defines bifurcation. It also offers some methods that can describe the bifurcations well,
        e.g. Strahler number that describes branching.
        :param points:
        """
        if not isinstance(points, list):
            raise ValueError('Init bifurcation with ', type(points))
        # points are sorted from the biggest scale
        if np.any((np.diff([p[0] for p in points]) > 0)):
            raise Exception('Rows must be in decreasing order, but they are: ', points)

        self.points = points
        self.children = []

    def __str__(self):
        parent = ', '.join(map(str, self.points[:3])) + '\n'
        for ch in self.children:
            parent += '\t' + str(ch)
        return parent

    def __eq__(self, other):
        res = equal_points(self.points, other.points)
        if not res:
            return False
        else:
            for ch1, ch2 in zip(self.children, other.children):
                res = res and ch1 == ch2
                if not res:
                    return False
            return res

    def is_merged(self):
        return len(self.children) > 0

    def get_b(self, a):
        if self.points[-1][0] > a:
            return self.children[-1].get_b(a)
        else:
            found_b = [bb for aa, bb in self.points if a == aa]
            return found_b[0]

    def add_child(self, child):
        if not isinstance(child, Bifurcation):
            raise AssertionError
        lowest_a = self.points[-1][0]
        childs_max_a = child.points[0][0]
        if childs_max_a > lowest_a:
            print('WARNING: add_child()')
            print('parents lowest_a: \t',  lowest_a)
            print('childs max_a: \t', childs_max_a)
            raise AssertionError()
        self.children.append(child)

    def set_child(self, new_child, index):
        if not isinstance(new_child, Bifurcation):
            raise AssertionError
        lowest_a = self.points[-1][0]
        childs_max_a = new_child.points[0][0]
        if childs_max_a > lowest_a:
            print('WARNING: set_child()')
            print('parents lowest_a: \t',  lowest_a)
            print('childs max_a: \t', childs_max_a)
            raise AssertionError()
        self.children[index] = new_child

    def get_points(self):
        res = copy.deepcopy(self.points)
        for ch in self.children:
            res.extend(ch.get_points())
        return res

    def print_points(self):
        res = self.points
        print(res)
        print()
        for ch in self.children:
            print('.')
            ch.print_points()

    def get_level(self):
        if len(self.children) == 0:
            return 1
        lvls = [ch.get_level() for ch in self.children]
        lvl = np.max(lvls) + 1
        return lvl

    def get_strahler_nr(self):
        if len(self.children) == 0:
            return 1
        str_nrs = [ch.get_strahler_nr() for ch in self.children]

        str_nrs.sort(reverse=True)
        s1, s2 = str_nrs[:2]
        if s1 > s2:
            res = s1
        elif s1 == s2:
            res = s1 + 1
        else:
            raise Exception('sorting is wrong')
        return res

    def get_left_edge(self):
        if len(self.children) == 0:
            return self.points[-1][1]
        else:
            return self.children[0].get_left_edge()

    def get_right_edge(self):
        if len(self.children) == 0:
            return self.points[-1][1]
        else:
            return self.children[-1].get_right_edge()

    def get_spread(self):
        l = self.get_left_edge()
        r = self.get_right_edge()
        spread = r - l
        assert r >= l
        return l, spread

    def get_n_branches_at_half_and_qrt(self, verbose=False):
        """
        Returns number of branches at half and quarter of the scale
        :return:
        """
        a_max = self.points[0][0]
        a_half = int(a_max/2.0)
        a_qrt = int(a_max/4.0)
        if verbose:
            print('a_half: ', a_half)
            print('a_qrt: ', a_qrt)
        all_points = self.get_points()
        points_at_half = [(a, b) for a, b in all_points if a == a_half]
        points_at_qrt = [(a, b) for a, b in all_points if a == a_qrt]
        n_at_half = len(points_at_half)
        n_at_qrt = len(points_at_qrt)
        if verbose:
            print('points_at_half: ', points_at_half)
            print('points_at_qrt: ', points_at_qrt)

        if n_at_qrt < n_at_half or n_at_half < 1:
            raise AssertionError
        return n_at_half, n_at_qrt

    @staticmethod
    def merge(bif1, bif2):
        # assert that both bif1 and bif2 are whole, but bif2 must be unmerged yet
        # case 1: merging happens in the bif1.points
        # case 2: merging happens in bif1.right
        # case 3: merging not possible

        bif1 = copy.deepcopy(bif1)
        bif2 = copy.deepcopy(bif2)

        assert not bif2.is_merged()

        ind1, ind2 = equal_pt_indices(bif1.points, bif2.points, reverse=True)

        if -1 < ind1 < len(bif1.points) - 1:
            # print('Case 1: merging happens in the bif1.points')
            # case 1: we can merge points directly
            common_points = bif1.points[:ind1 + 1]
            bif1.points = bif1.points[ind1 + 1:]
            bif2.points = bif2.points[ind2 + 1:]
            assert len(bif1.points) > 0 and len(bif2.points) > 0
            if bif1.points[0][0] != bif2.points[0][0]:
                print(bif1.points)
                print(bif2.points)
                raise AssertionError('First Branched point must have the same scales')
            # compare column of the first different point to decide which one is left
            if bif1.points[0][1] < bif2.points[0][1]:
                left = bif1
                right = bif2
            elif bif1.points[0][1] > bif2.points[0][1]:
                # left = bif2
                # right = bif1
                raise AssertionError('bif1 should always be on the left side')
            else:
                print(bif1.points)
                print(bif2.points)
                raise AssertionError('The separated lines have the same column, bt it must be different.')

            new_bif = Bifurcation(common_points)
            #new_bif.children.append(left)
            #new_bif.children.append(right)
            new_bif.add_child(left)
            new_bif.add_child(right)
            return True, new_bif

        elif ind1 == len(bif1.points) - 1:
            # print('Case 2: merging might happen in bif1.right')
            if len(bif1.children) < 2 and bif1.children[-1] is None:
                print(bif1.points)
                print(bif2.points)
                raise AssertionError('Common elements go up to the begin of .points, but no .right exists for bif1')
            success, merged = Bifurcation.merge(bif1.children[-1], bif2)
            if success:
                #bif1.children[-1] = merged
                bif1.set_child(merged, -1)
                return True, bif1
            else:
                # print('This might be a special case with multi branching')
                third_child = Bifurcation(bif2.points[ind2 + 1:])
                #bif1.children.append(third_child)
                bif1.add_child(third_child)
                return True, bif1

        else:
            # print('Case 3: merging not possible')
            return False, None


def best_direction(hood, center_col):
    """
    For a given region of space in the wavelet coefficient matrix (the neighborhood) find the non-zero points,
    make them into a list, and sort them according to their net distance.

    :param hood: neighborhood to look in, a subset of the wtmm mask
    :param center_col: location of the current point in the hood's columns
    :return: the best match (head of the list)
    """
    nzs = np.where(hood > 0)

    # tuple of (abs-dist, value, (row, col))
    ref_r, ref_c = -1, center_col
    closest_points = [(np.sqrt(np.square(ref_r-r) + np.square(ref_c-c)), (r, c)) for r, c in zip(*nzs)]
    if len(closest_points) > 0:
        closest_points.sort(key=itemgetter(0))
        return closest_points[0]
    else:
        return None


def walk_bifurcation(mtx, start_col, proximity):
    """
    For a given wtmm mask, derive a contiguous line for any given starting point. Starting point is the row 0
    (i.e. smallest scale) and and col = start_col.

    :param mtx: WTMM mask
    :param start_col: column of the starting point
    :param proximity: how far this function should look in the vicinity of the current point.
    :return: tuple(bool, list of coordinates) - bool for if the line hits the ground, coordinates of the points consumed
    :raise ValueError:
    """
    if start_col < 0 or start_col >= mtx.shape[1]:
        raise ValueError('start_col is out of bounds for matrix shape {}: {}'.format(mtx.shape, start_col))
    if proximity > int((mtx.shape[1] - 1) / 2.0):
        raise ValueError('proximity is too big for matrix of shape {}: {}'.format(mtx.shape, proximity))

    center_row, center_col = 0, start_col
    max_row, max_col = [i - 1 for i in mtx.shape]
    trace_rt = [(center_row, center_col)]

    while center_row < mtx.shape[0] - 1:

        # get the proximity bounds for a given point in the matrix (addresses to look in)
        right_bound = center_col + proximity + 1
        left_bound = center_col - proximity
        hood_center_col = proximity
        if right_bound > max_col:
            right_bound = max_col
        elif left_bound < 0:
            hood_center_col = proximity + left_bound
            assert hood_center_col >= 0
            left_bound = 0

        upper_bound = center_row + proximity
        if upper_bound >= mtx.shape[0]:
            upper_bound = mtx.shape[0] - 1

        # get the neighborhood of addresses...
        assert left_bound >= 0 and right_bound < mtx.shape[1]
        hood = mtx[center_row + 1:upper_bound, left_bound:right_bound]

        # find the best choice for the ridge
        closest = best_direction(hood, center_col=hood_center_col)

        if closest is None:
            # Means we've reached the end of the ridge line
            if len(trace_rt) == 0:
                return False, trace_rt
            else:
                return True, trace_rt

        # recompute the center of the addresses and continue
        dist, (match_hood_row, match_hood_col) = closest

        # match_hood_row < proximity always (this moves us up the matrix rows) but is always off by 1
        center_row += 1 + match_hood_row
        # this can be +/- depending on the direction
        center_col += match_hood_col - hood_center_col

        assert 0 <= center_col < mtx.shape[1]

        trace_rt.append((center_row, center_col))
        #print(trace_rt)

        if center_col == max_col or center_col == 0:
            # If we end up on and edge, this is not a valid bifurcation
            return False, trace_rt

    return True, trace_rt


def skeletor(input_mtx, proximity=9, plot=False, scales=None):
    """
    Skeleton Constructor

    The basic ideas is to scan the coefficient matrix from row 0 to row n-1 looking for non-zero elements. It assumes
    that the matrix has already been cleaned of everything that is not a local maxima. I generally use order=1 for this.

    :param input_mtx: WTMM mask
    :param proximity: how near by a non-zero point to look for the next non-zero point to jump to.
    :param plot: plot the skeleton that is constructed
    :param scales: list of scales. Here used ONLY because of plotting.
    :returns list of bifurcation objects after merging
    """
    # to avoid side-effect, we will work on the matrix copy
    mtx = input_mtx.copy()

    # NB: scale <-> row
    # NB: shift <-> col
    max_row, max_col = mtx.shape
    max_row -= 1
    max_col -= 1

    # holder for the ridges
    bifurcations = OrderedDict()
    invalids = OrderedDict()
    bi_cnt = 0

    # local maxima at the lowest scale
    maxs = signal.argrelmax(mtx[0])[0]

    for start_pt in maxs:
        continuous, bifurc_path = walk_bifurcation(mtx, start_col=start_pt, proximity=proximity)

        if continuous:
            # add the bifurcation to the collector; key == row[0] intercept's column number
            bifurcations[(bi_cnt, bifurc_path[-1][1])] = bifurc_path
            bi_cnt += 1
        elif bifurc_path:
            invalids[bifurc_path[-1]] = bifurc_path

    # interpolate the ridge lines (i.e. fill the missing values). The path gets reversed here.
    for k, v in bifurcations.items():
        v = v[::-1]
        rows, cols = zip(*v)
        rows = rows[::-1]
        cols = cols[::-1]
        if np.isscalar(k[-1]):
            missing_rows = list(set(np.arange(np.max(rows) + 1)).difference(set(rows)))

            if np.any((np.diff(rows) < 0)):
                raise Exception('Rows must be in increasing order, but they are: ', rows)

            missing_cols = np.interp(missing_rows, rows, cols)
            new_rows = np.concatenate((rows, missing_rows)).astype(int)
            new_cols = np.concatenate((cols, missing_cols)).astype(int)
            new_value = list(zip(new_rows, new_cols))
            new_value.sort(key=lambda e: e[0], reverse=True)
            bifurcations[k] = new_value

    assert len(bifurcations) > 0

    bif_objects = []
    for k, v in bifurcations.items():
        bif_objects.append(Bifurcation(v))

    # connect the ridge lines that have common points
    bif_objects.sort(key=lambda x: x.points[-1][1])
    final_list = []
    while len(bif_objects) > 1:
        l = bif_objects.pop(0)
        r = bif_objects.pop(0)
        assert not r.is_merged()
        success, merged = Bifurcation.merge(l, r)
        if success:
            bif_objects.insert(0, merged)
        else:
            final_list.append(l)
            bif_objects.insert(0, r)

    final_list.append(bif_objects[0])

    if plot:
        plt.figure(figsize=(10, 8))
        #plt.title('Bifurcations')
        colors = ['C0', 'C1', 'C2', 'C3',  'C4', 'C5', 'C6', 'C7']
        for n, bif in enumerate(final_list):
            points = bif.get_points()
            rows, cols = zip(*points)
            plt.plot(cols, rows, 'o', color= colors[n % len(colors)], alpha=0.9)
            #print(bif.get_strahler_nr())
            # break

        if not np.all(np.diff(scales) == 1):
            scales_str = ['%.2f' % sc for sc in scales]
            plt.yticks(range(mtx.shape[0]), scales_str)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.xlabel('Dilation b')
        plt.ylabel('Scale a')
        plt.title('Bifurcations')
        plt.show()

    final_list.sort(key=lambda x: x.points[0][0], reverse=True)

    return final_list
