# SPDX-License-Identifier: MIT
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
import route
import check_result


class RoutingTests(unittest.TestCase):
    def test_lease_and_design_only_prs_skip_native_suites(self):
        paths = ['docs/ROADMAP.md', 'docs/roadmap/tasks/F8-05.json',
                 'docs/roadmap/F8-05-SHARED-SPATIAL-INDEX.md',
                 'tests/roadmap/test_f8_spatial_design.py']
        self.assertFalse(any(route.classify(paths).values()))

    def test_ordinary_prose_and_roadmap_tool_run_contracts_only(self):
        for path in ('README.md', 'AGENTS.md', 'CONTRIBUTING.md', 'LICENSE',
                     'docs/architecture.md', 'docs/history/ROADMAP-old.md',
                     'tools/roadmap.py', '.github/pull_request_template.md'):
            with self.subTest(path=path):
                self.assertFalse(any(route.classify([path]).values()))

    def test_reference_artifacts_are_not_skipped_as_docs(self):
        for path in ('docs/reference/README.md', 'docs/reference/api/parallel_mater.h',
                     'docs/reference/parallel-mater-reference.pdf',
                     'tools/reference/build.py', 'tests/release/test_reference_artifacts.py'):
            with self.subTest(path=path):
                self.assertEqual(route.classify([path]),
                                 {s: s == 'reference' for s in route.SUITES})

    def test_code_build_ci_unknown_and_mixed_changes_run_every_suite(self):
        for path in ('src/slang/fluid.slang', 'src/driver/vulkan/schedule_context.cpp',
                     'include/parallel_mater/parallel_mater.h', 'registry/pm.xml',
                     'CMakeLists.txt', 'CMakePresets.json', 'cmake/SlangKernelManifest.cmake',
                     '.github/workflows/sdk-install.yml', 'tools/ci/route.py',
                     'tests/ci/test_routing.py', 'tests/sdk/compatibility.c',
                     'examples/android/app/build.gradle.kts', 'new-directory/new-input',
                     'unknown.md', 'docs/new-machine-readable-input.json'):
            with self.subTest(path=path):
                self.assertTrue(all(route.classify(['docs/ROADMAP.md', path]).values()))

    def test_force_full_includes_doc_only_and_empty_diffs(self):
        self.assertTrue(all(route.classify(['docs/ROADMAP.md'], full=True).values()))
        self.assertTrue(all(route.classify([], full=True).values()))
        self.assertFalse(any(route.classify([]).values()))

    def test_scheduled_manual_and_unknown_events_run_full(self):
        for name in ('schedule', 'workflow_dispatch', 'unexpected'):
            self.assertTrue(all(route.plan_for_event(name, {})[0].values()))

    def test_missing_malformed_and_zero_commits_fail_to_full(self):
        for name, event in (('pull_request', {}), ('pull_request', []),
                            ('push', {'before': '0'*40, 'after': 'a'*40}),
                            ('push', {'before': '--help', 'after': 'b'*40}),
                            ('merge_group', {'merge_group': {}})):
            with self.subTest(name=name, event=event):
                plan, reason = route.plan_for_event(name, event)
                self.assertTrue(all(plan.values()))
                self.assertIn('diff unavailable', reason)

    def test_failed_git_diff_selects_full_not_empty(self):
        with patch.object(route, 'changed_paths', side_effect=subprocess.CalledProcessError(1, 'git')):
            plan, reason = route.plan_for_event('push', {'before': 'a'*40, 'after': 'b'*40})
        self.assertTrue(all(plan.values()))
        self.assertIn('diff unavailable', reason)

    def test_event_ranges_use_pr_merge_base_but_direct_push_and_queue_ranges(self):
        cases = [
            ('pull_request', {'pull_request': {'base': {'sha': 'a'*40}, 'head': {'sha': 'b'*40}}}, True),
            ('push', {'before': 'a'*40, 'after': 'b'*40}, False),
            ('merge_group', {'merge_group': {'base_sha': 'a'*40, 'head_sha': 'b'*40}}, False),
        ]
        for name, event, merge_base in cases:
            with self.subTest(name=name), patch.object(route, 'changed_paths', return_value=['docs/ROADMAP.md']) as diff:
                plan, _ = route.plan_for_event(name, event, cwd='/test')
                self.assertFalse(any(plan.values()))
                kwargs = {'cwd': '/test'} | ({'merge_base': True} if merge_base else {})
                diff.assert_called_once_with('a'*40, 'b'*40, **kwargs)

    def test_cli_writes_boolean_outputs_and_summary_for_docs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event, output, summary = (root / name for name in ('event.json', 'output', 'summary'))
            event.write_text('{}')
            argv = ['route', '--event-name', 'pull_request', '--event-path', str(event),
                    '--output', str(output), '--summary', str(summary)]
            with patch.object(sys, 'argv', argv), patch.object(route, 'plan_for_event',
                    return_value=(route.classify([]), 'fixture')):
                route.main()
            lines = dict(line.split('=', 1) for line in output.read_text().splitlines())
            self.assertEqual(json.loads(lines['plan']), route.classify([]))
            self.assertTrue(all(lines[s] == 'false' for s in route.SUITES))
            self.assertIn('not platform execution', summary.read_text())


class GitDiffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-b', 'main')

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'user.name=CI Fixture',
                '-c', 'user.email=ci-fixture@example.invalid', *args], cwd=self.root,
                stderr=subprocess.PIPE, text=True).strip()

    def write(self, name, text='fixture\n'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-m', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def test_complete_diff_exceeds_hosted_path_filter_limits_and_preserves_names(self):
        self.write('README.md')
        base = self.commit()
        for i in range(3101):
            self.write(f'docs/roadmap/fixture-{i}.md')
        unusual = 'src/space and\nnewline-λ.cpp'
        self.write(unusual)
        head = self.commit()
        paths = route.changed_paths(base, head, cwd=self.root)
        self.assertEqual(len(paths), 3102)
        self.assertIn(unusual, paths)
        self.assertTrue(all(route.classify(paths).values()))

    def test_rename_and_deletion_keep_original_code_path(self):
        self.write('src/code.cpp')
        self.write('src/deleted.cpp')
        base = self.commit()
        self.git('mv', 'src/code.cpp', 'README.md')
        (self.root / 'src/deleted.cpp').unlink()
        head = self.commit()
        paths = route.changed_paths(base, head, cwd=self.root)
        self.assertEqual(set(paths), {'src/code.cpp', 'src/deleted.cpp', 'README.md'})
        self.assertTrue(all(route.classify(paths).values()))

    def test_pr_uses_merge_base_not_unrelated_new_base_changes(self):
        self.write('README.md')
        self.commit()
        self.git('checkout', '-b', 'topic')
        self.write('docs/roadmap/plan.md')
        head = self.commit()
        self.git('checkout', 'main')
        self.write('src/unrelated.cpp')
        base = self.commit()
        paths = route.changed_paths(base, head, merge_base=True, cwd=self.root)
        self.assertEqual(paths, ['docs/roadmap/plan.md'])


def results(plan):
    return {'plan': {'result': 'success', 'outputs': {'plan': json.dumps(plan)}},
            **{s: {'result': 'success' if enabled else 'skipped'} for s, enabled in plan.items()}}


class ResultGateTests(unittest.TestCase):
    def test_full_docs_and_reference_success(self):
        for paths in (['src/physics.cpp'], ['docs/ROADMAP.md'], ['docs/reference/api-coverage.json']):
            check_result.check(results(route.classify(paths)))

    def test_selected_suite_skipped_cancelled_failed_or_missing_never_passes(self):
        for suite in route.SUITES:
            for result in ('skipped', 'cancelled', 'failure', ''):
                needs = results(route.classify([], full=True))
                needs[suite]['result'] = result
                with self.subTest(suite=suite, result=result), self.assertRaises(ValueError):
                    check_result.check(needs)

    def test_routing_failure_or_cancellation_cannot_turn_into_green_skips(self):
        for state in ('failure', 'cancelled', 'skipped'):
            needs = results(route.classify([]))
            needs['plan']['result'] = state
            with self.assertRaises(ValueError):
                check_result.check(needs)

    def test_missing_job_output_and_wrong_boolean_types_fail(self):
        valid = results(route.classify([]))
        variants = [None, [], {}, {k: v for k, v in valid.items() if k != 'mobile'}]
        for output in ('{}', 'null', 'invalid-json', json.dumps({s: 'false' for s in route.SUITES})):
            bad = copy.deepcopy(valid)
            bad['plan']['outputs']['plan'] = output
            variants.append(bad)
        for needs in variants:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                check_result.check(needs)

    def test_unselected_suite_cannot_run_and_hide_failure(self):
        needs = results(route.classify([]))
        needs['mobile']['result'] = 'failure'
        with self.assertRaises(ValueError):
            check_result.check(needs)


if __name__ == '__main__':
    unittest.main()
