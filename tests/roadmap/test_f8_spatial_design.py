# SPDX-License-Identifier: MIT
"""Independent design model, NOT production BVH/solver qualification.

Uses a simple median tree, not the future GPU builder. The assertion is that
conservative envelopes and the proposed epoch rules include required work.
"""
from dataclasses import dataclass
import math
import random
import unittest


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


@dataclass(frozen=True)
class Box:
    low: tuple
    high: tuple

    def __post_init__(self):
        if (len(self.low) != 3 or len(self.high) != 3
                or not all(math.isfinite(v) for v in self.low + self.high)
                or any(a > b for a, b in zip(self.low, self.high))):
            raise ValueError('invalid bounds')

    def expanded(self, amount):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('invalid expansion')
        return Box(tuple(v - amount for v in self.low),
                   tuple(v + amount for v in self.high))

    def overlaps(self, other):
        return all(a <= d and c <= b for a, b, c, d in
                   zip(self.low, self.high, other.low, other.high))

    def contains(self, point):
        return (len(point) == 3 and all(math.isfinite(p) for p in point)
                and all(a <= p <= b for a, p, b in zip(self.low, point, self.high)))


def hull(points):
    points = list(points)
    if not points or any(len(p) != 3 or not all(math.isfinite(v) for v in p)
                         for p in points):
        raise ValueError('empty or nonfinite hull')
    return Box(tuple(min(p[i] for p in points) for i in range(3)),
               tuple(max(p[i] for p in points) for i in range(3)))


def union(boxes):
    return hull(p for b in boxes for p in (b.low, b.high))


def envelope(start, predicted, path_slack=0., correction=0., radius=0.):
    # Scalar model uses the same bound on all axes. GPU outward rounding is a
    # separate implementation obligation; this CPU model is exact-real intent.
    if any(not math.isfinite(v) or v < 0
           for v in (path_slack, correction, radius)):
        raise ValueError('invalid envelope budget')
    return hull((start, predicted)).expanded(path_slack + correction + radius)


def weighted_box(boxes, weights):
    if (len(boxes) != len(weights) or not boxes
            or not all(math.isfinite(w) for w in weights)):
        raise ValueError('invalid binding')
    return Box(tuple(sum(min(w*b.low[i], w*b.high[i])
                         for w, b in zip(weights, boxes)) for i in range(3)),
               tuple(sum(max(w*b.low[i], w*b.high[i])
                         for w, b in zip(weights, boxes)) for i in range(3)))


@dataclass(frozen=True, order=True)
class Identity:
    family: int
    owner_slot: int
    owner_generation: int
    topology_revision: int
    primitive_kind: int
    primitive_slot: int
    primitive_generation: int


def identity(slot, owner=0, owner_generation=1, generation=1):
    return Identity(1, owner, owner_generation, 1, 0, slot, generation)


@dataclass(frozen=True)
class Node:
    bounds: Box
    identity: object = None
    left: object = None
    right: object = None


def build(entries):
    if not entries:
        return None
    if len({key for key, _ in entries}) != len(entries):
        raise ValueError('duplicate identity')
    if len(entries) == 1:
        key, bounds = entries[0]
        return Node(bounds, identity=key)
    ordered = sorted(entries, key=lambda item:
                     (item[1].low[0] + item[1].high[0], item[0]))
    middle = len(ordered) // 2
    left, right = build(ordered[:middle]), build(ordered[middle:])
    return Node(union((left.bounds, right.bounds)), left=left, right=right)


def query(root, bounds, maximum_visits=None):
    result, pending, visits = set(), [root] if root else [], 0
    while pending:
        visits += 1
        if maximum_visits is not None and visits > maximum_visits:
            raise OverflowError('query work budget exhausted; no partial success')
        node = pending.pop()
        if not node.bounds.overlaps(bounds):
            continue
        if node.identity is not None:
            result.add(node.identity)
        else:
            pending.extend((node.left, node.right))
    return result


def admit(live, proposals, spacing):
    """Proposed one-pass priority rule; intentionally not a greedy packing."""
    return {key for key, point in proposals.items()
            if all(math.dist(point, other) >= spacing for other in live.values())
            and all(other_key >= key or math.dist(point, other) >= spacing
                    for other_key, other in proposals.items())}


def boundary_transfer(state, source, destination, destination_capacity):
    """Atomic identity replacement model; does not model thermal/force physics."""
    target_owner = (destination.family, destination.owner_slot, destination.owner_generation)
    count = sum((k.family, k.owner_slot, k.owner_generation) == target_owner for k in state)
    if source not in state or destination in state or count >= destination_capacity:
        return dict(state), False
    staged = dict(state)
    staged[destination] = staged.pop(source)
    return staged, True


class EpochModel:
    """Transactional state model; actual GPU rollback/ordering is still required."""
    def __init__(self, state):
        self.committed = dict(state)
        self.tick = 0
        self.update_count = 0
        self.publication_count = 0
        self.pending = None
        self.bounds = None
        self.tree = None
        self.retained = set()
        self.transfers = {}
        self.published = False
        self.failed = False

    def prepare(self, pending, bounds, transfers=None):
        if self.pending is not None:
            raise RuntimeError('tick already prepared')
        if set(pending) != set(bounds):
            raise ValueError('bounds must cover every staged identity')
        if any(not bounds[k].contains(p) for k, p in pending.items()):
            raise ValueError('staged state outside envelope')
        transfers = dict(transfers or {})
        if (len(set(transfers.values())) != len(transfers)
                or any(source not in self.committed or source not in pending
                       or destination not in pending or destination in self.committed
                       for source, destination in transfers.items())):
            raise ValueError('invalid conditional transfer')
        self.pending, self.bounds = dict(pending), dict(bounds)
        # Planned removals already happened at the boundary; admission may
        # reject only new proposals, not the surviving committed identities.
        self.retained = set(self.committed) & set(pending)
        self.transfers = transfers
        self.update_count = self.publication_count = 0
        self.published = False
        self.failed = False

    def update(self):
        if self.pending is None or self.update_count:
            raise RuntimeError('missing preparation or second update')
        self.tree = build(list(self.bounds.items()))
        self.update_count = 1

    def publish(self, admitted=None):
        if self.update_count != 1 or self.published:
            raise RuntimeError('invalid publication')
        allowed = set(self.pending) if admitted is None else set(admitted)
        if not allowed <= set(self.pending):
            raise ValueError('birth after build')
        if any((source in allowed) == (destination in allowed)
               for source, destination in self.transfers.items()):
            raise ValueError('transfer must select source XOR destination')
        retired = {source for source, destination in self.transfers.items()
                   if destination in allowed}
        if not self.retained <= allowed | retired:
            raise ValueError('admission cannot remove retained state')
        self.pending = {k: v for k, v in self.pending.items() if k in allowed}
        self.publication_count = 1
        self.published = True

    def candidates(self, bounds, maximum_visits=None):
        if not self.published or self.failed:
            raise RuntimeError('unpublished or failed tick')
        try:
            return query(self.tree, bounds, maximum_visits) & set(self.pending)
        except OverflowError:
            self.failed = True
            raise

    def solve(self, proposed):
        if self.failed:
            raise RuntimeError('failed tick needs abort')
        if not self.published or set(proposed) != set(self.pending):
            self.failed = True
            raise RuntimeError('stale epoch or mid-solve identity mutation')
        if any(not self.bounds[k].contains(p) for k, p in proposed.items()):
            self.failed = True
            raise ValueError('motion envelope escaped')
        self.pending = dict(proposed)

    def commit(self):
        if not self.published or self.failed:
            raise RuntimeError('unpublished or failed tick')
        self.committed = dict(self.pending)
        self.tick += 1
        self.pending = None
        self.published = False

    def abort(self):
        self.pending = None
        self.published = False


class SpatialEnvelopeTests(unittest.TestCase):
    def test_random_full_tick_neighbors_are_in_tree_candidates(self):
        rng = random.Random(0xF805)
        for _ in range(20):
            starts = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(20)]
            ends = [tuple(v + rng.uniform(-1, 1) for v in p) for p in starts]
            correction, slack = .15, .1
            supports = [rng.uniform(.1, .7) for _ in starts]
            boxes = [envelope(a, b, slack, correction) for a, b in zip(starts, ends)]
            tree = build([(identity(i), b) for i, b in enumerate(boxes)])
            for t in (0., .2, .5, .8, 1.):
                positions = [tuple(v + rng.uniform(-slack-correction, slack+correction)
                                   for v in lerp(a, b, t)) for a, b in zip(starts, ends)]
                for i, position in enumerate(positions):
                    candidates = query(tree, boxes[i].expanded(max(supports)))
                    actual = {identity(j) for j, other in enumerate(positions)
                              if math.dist(position, other) <= max(supports[i], supports[j])}
                    self.assertLessEqual(actual, candidates)

    def test_endpoint_only_rotation_is_a_failing_negative_control(self):
        radius = 2.
        start, end = (radius, 0., 0.), (radius, 0., 0.)  # A complete turn.
        halfway_quarter = (0., radius, 0.)
        naive = hull((start, end)).expanded(.01)
        conservative = envelope((0., 0., 0.), (0., 0., 0.), radius=radius)
        self.assertFalse(naive.contains(halfway_quarter))
        for step in range(361):
            angle = step * math.pi / 180
            self.assertTrue(conservative.contains(
                (radius * math.cos(angle), radius * math.sin(angle), 0.)))

    def test_rotating_translating_body_with_offset_mesh_center(self):
        a, b, center, radius = (-1., 0., 1.), (2., 3., -1.), 3., .5
        bounds = envelope(a, b, correction=.1, radius=center+radius)
        for step in range(101):
            t = step / 100
            position = lerp(a, b, t)
            angle = 4 * math.pi * t
            vertex = (position[0] + (center+radius)*math.cos(angle),
                      position[1] + (center+radius)*math.sin(angle), position[2])
            self.assertTrue(bounds.contains(vertex))

    def test_tighter_rotation_bound_uses_total_angular_travel(self):
        radius = 2.
        initial_translated = hull(((-radius, 0., 0.), (radius + 3., 0., 0.)))
        for travel in (0., .1, math.pi / 2, math.pi, 2*math.pi, 4*math.pi):
            bounds = initial_translated.expanded(min(radius * travel, 2 * radius) + 1e-12)
            for step in range(101):
                t = step / 100
                for sign in (-1, 1):
                    point = (3*t + sign*radius*math.cos(travel*t),
                             sign*radius*math.sin(travel*t), 0.)
                    self.assertTrue(bounds.contains(point))

    def test_segment_and_triangle_envelopes_cover_interiors(self):
        starts = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
        ends = [(1., 0., 1.), (2., 1., 0.), (-1., 2., 1.)]
        boxes = [envelope(a, b, correction=.1) for a, b in zip(starts, ends)]
        triangle, segment = union(boxes), union(boxes[:2])
        for t in (0., .25, .5, 1.):
            points = [lerp(a, b, t) for a, b in zip(starts, ends)]
            for u in (0., .25, .5, 1.):
                self.assertTrue(segment.contains(lerp(points[0], points[1], u)))
                self.assertTrue(triangle.contains(
                    lerp(lerp(points[0], points[1], u), points[2], .3)))

    def test_signed_skin_binding_requires_interval_weights(self):
        boxes = [hull(((0., 0., 0.), (1., 1., 1.))),
                 hull(((2., 2., 2.), (3., 3., 3.)))]
        point = (-3., -3., -3.)  # 2 * first.low - second.high.
        self.assertFalse(union(boxes).contains(point))
        self.assertTrue(weighted_box(boxes, (2., -1.)).contains(point))

    def test_touching_empty_and_duplicate_geometry(self):
        unit = hull(((0., 0., 0.), (1., 1., 1.)))
        touching = hull(((1., 1., 1.), (2., 2., 2.)))
        tree = build([(identity(0), unit), (identity(1), unit)])
        self.assertEqual(query(tree, touching), {identity(0), identity(1)})
        self.assertEqual(query(build([]), unit), set())
        with self.assertRaisesRegex(ValueError, 'duplicate identity'):
            build([(identity(0), unit), (identity(0), touching)])

    def test_nonfinite_negative_and_traversal_overflow_reject(self):
        for a, b in (((math.nan, 0., 0.), (0., 0., 0.)),
                     ((0., 0., 0.), (math.nan, 0., 0.)),
                     ((0., 0., 0.), (math.inf, 0., 0.))):
            with self.assertRaises(ValueError):
                envelope(a, b)
        with self.assertRaises(ValueError):
            envelope((0., 0., 0.), (0., 0., 0.), correction=-1.)
        unit = hull(((0., 0., 0.), (1., 1., 1.)))
        tree = build([(identity(0), unit), (identity(1), unit)])
        with self.assertRaisesRegex(OverflowError, 'no partial success'):
            query(tree, unit, maximum_visits=1)

    def test_malformed_point_and_nonfinite_binding_reject(self):
        unit = hull(((0., 0., 0.), (1., 1., 1.)))
        for point in ((), (0.,), (0., 0., 0., 0.), (math.nan, 0., 0.)):
            self.assertFalse(unit.contains(point))
        for weight in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                weighted_box([unit], [weight])


class SpatialEpochTests(unittest.TestCase):
    def test_identity_survives_compaction_and_distinguishes_reuse(self):
        a, b = identity(0), identity(0, owner=1)
        old_owner, reused_particle = identity(0, owner_generation=2), identity(0, generation=2)
        self.assertEqual(len({a, b, old_owner, reused_particle}), 4)
        original = {a: 7, b: 1}
        frozen = dict(original)
        original[a], original[b] = 0, 2
        self.assertEqual(frozen, {a: 7, b: 1})
        self.assertNotIn(reused_particle, frozen)

    def test_one_publication_reuses_tree_through_iterations(self):
        key = identity(0)
        epoch = EpochModel({key: (0., 0., 0.)})
        epoch.prepare(epoch.committed, {key: envelope((0., 0., 0.), (1., 0., 0.), correction=.1)})
        epoch.update()
        tree = epoch.tree
        epoch.publish()
        for x in (.2, .7, 1.05):
            epoch.solve({key: (x, 0., 0.)})
            self.assertIs(tree, epoch.tree)
        with self.assertRaises(RuntimeError):
            epoch.update()
        with self.assertRaises(RuntimeError):
            epoch.publish()
        epoch.commit()
        self.assertEqual((epoch.tick, epoch.update_count, epoch.publication_count), (1, 1, 1))

    def test_escape_and_birth_after_publication_cannot_change_committed_state(self):
        key = identity(0)
        for proposed, error in (({key: (1., 0., 0.), identity(1): (0., 0., 0.)}, RuntimeError),
                                ({key: (2., 0., 0.)}, ValueError)):
            epoch = EpochModel({key: (0., 0., 0.)})
            original = dict(epoch.committed)
            epoch.prepare(original, {key: envelope((0., 0., 0.), (1., 0., 0.))})
            epoch.update()
            epoch.publish()
            epoch.solve({key: (.5, 0., 0.)})
            with self.assertRaises(error):
                epoch.solve(proposed)
            with self.assertRaisesRegex(RuntimeError, 'failed tick'):
                epoch.commit()
            epoch.abort()
            self.assertEqual((epoch.committed, epoch.tick), (original, 0))

    def test_correction_budget_is_total_not_per_iteration(self):
        key, start = identity(0), (0., 0., 0.)
        epoch = EpochModel({key: start})
        epoch.prepare(epoch.committed, {key: envelope(start, start, correction=1.)})
        epoch.update()
        epoch.publish()
        epoch.solve({key: (.4, 0., 0.)})
        epoch.solve({key: (.8, 0., 0.)})
        with self.assertRaises(ValueError):
            epoch.solve({key: (1.2, 0., 0.)})
        with self.assertRaises(RuntimeError):
            epoch.commit()
        self.assertEqual(epoch.committed[key], start)

    def test_query_overflow_invalidates_tick_not_just_query_result(self):
        state = {identity(0): (0., 0., 0.), identity(1): (1., 0., 0.)}
        bounds = {k: envelope(p, p) for k, p in state.items()}
        epoch = EpochModel(state)
        epoch.prepare(state, bounds)
        epoch.update()
        epoch.publish()
        with self.assertRaises(OverflowError):
            epoch.candidates(union(bounds.values()), maximum_visits=1)
        with self.assertRaises(RuntimeError):
            epoch.commit()
        self.assertEqual((epoch.committed, epoch.tick), (state, 0))

    def test_invalid_preparation_and_retained_leaf_removal_reject(self):
        key, start = identity(0), (0., 0., 0.)
        epoch = EpochModel({key: start})
        with self.assertRaises(ValueError):
            epoch.prepare({key: (1., 0., 0.)}, {key: envelope(start, start)})
        epoch.prepare(epoch.committed, {key: envelope(start, start)})
        epoch.update()
        with self.assertRaisesRegex(ValueError, 'retained state'):
            epoch.publish(set())
        self.assertEqual(epoch.publication_count, 0)
        self.assertEqual(epoch.committed, {key: start})

    def test_phase_transfer_waits_for_boundary_and_capacity(self):
        source = identity(0)
        destination = Identity(2, 0, 1, 1, 0, 0, 1)
        state = {source: (1., 2., 3.)}
        pending_request = (source, destination)
        self.assertIn(source, state)  # Recording a request does not remove mass.
        rejected, accepted = boundary_transfer(state, *pending_request, 0)
        self.assertFalse(accepted)
        self.assertEqual(rejected, state)
        transferred, accepted = boundary_transfer(state, *pending_request, 1)
        self.assertTrue(accepted)
        self.assertEqual(transferred, {destination: state[source]})
        self.assertEqual(len(transferred), len(state))
        self.assertEqual(state, {source: (1., 2., 3.)})  # Staging is not commit.

    def test_provisional_admission_uses_one_tree_with_inactive_leaves(self):
        live = {identity(0): (0., 0., 0.)}
        proposals = {identity(1): (.2, 0., 0.), identity(2): (2., 0., 0.),
                     identity(3): (2.2, 0., 0.)}
        allowed = admit(live, proposals, .5)
        self.assertEqual(allowed, {identity(2)})
        pending = live | proposals
        epoch = EpochModel(live)
        epoch.prepare(pending, {key: envelope(p, p) for key, p in pending.items()})
        epoch.update()
        tree = epoch.tree
        epoch.publish(set(live) | allowed)
        self.assertIs(tree, epoch.tree)
        whole = union(epoch.bounds.values())
        self.assertEqual(query(tree, whole), set(pending))  # Tombstones remain.
        self.assertEqual(epoch.candidates(whole), set(live) | allowed)
        epoch.commit()
        self.assertEqual(set(epoch.committed), set(live) | allowed)
        self.assertEqual(epoch.update_count, 1)

    def test_post_build_transfer_admission_preserves_one_side_without_rebuild(self):
        source, destination = identity(0), Identity(2, 0, 1, 1, 0, 0, 1)
        state = {source: (0., 0., 0.)}
        pending = state | {destination: (0., 0., 0.)}
        bounds = {k: envelope(p, p) for k, p in pending.items()}
        for selected in (source, destination):
            epoch = EpochModel(state)
            epoch.prepare(pending, bounds, transfers={source: destination})
            epoch.update()
            tree = epoch.tree
            for invalid in (set(), {source, destination}):
                with self.assertRaisesRegex(ValueError, 'XOR'):
                    epoch.publish(invalid)
            epoch.publish({selected})
            self.assertIs(epoch.tree, tree)
            self.assertEqual(epoch.candidates(bounds[source]), {selected})
            epoch.commit()
            self.assertEqual(set(epoch.committed), {selected})
            self.assertEqual((epoch.update_count, epoch.publication_count), (1, 1))

    def test_admission_order_independent_with_documented_underfill(self):
        proposals = {identity(0): (0., 0., 0.), identity(1): (.4, 0., 0.),
                     identity(2): (.8, 0., 0.)}
        self.assertEqual(admit({}, proposals, .5), {identity(0)})
        self.assertEqual(admit({}, dict(reversed(list(proposals.items()))), .5),
                         {identity(0)})  # A greedy rule could also admit slot 2.

    def test_empty_epoch_and_multiple_ticks(self):
        epoch = EpochModel({})
        for _ in range(2):
            epoch.prepare({}, {})
            epoch.update()  # Logical update only; empty build has no tree/work.
            self.assertIsNone(epoch.tree)
            epoch.publish()
            epoch.commit()
        self.assertEqual(epoch.tick, 2)


if __name__ == '__main__':
    unittest.main()
