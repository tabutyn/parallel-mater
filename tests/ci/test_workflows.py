# SPDX-License-Identifier: MIT
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
from route import SUITES


def workflow(name):
    # BaseLoader preserves GitHub's YAML 1.2 "on" key (not YAML 1.1 True).
    return yaml.load((ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader)


class WorkflowTests(unittest.TestCase):
    def test_one_automatic_entrypoint_no_branch_push_duplicate(self):
        automatic = {'pull_request', 'push', 'merge_group', 'schedule'}
        for path in (ROOT / '.github/workflows').glob('*.yml'):
            events = workflow(path.name)['on']
            if path.name == 'roadmap-leases.yml':
                self.assertEqual(set(events), automatic | {'workflow_dispatch'})
                self.assertEqual(events['push'], {'branches': ['main']})
                self.assertEqual(events['pull_request']['types'], ['opened', 'synchronize', 'reopened'])
            else:
                self.assertEqual(set(events), {'workflow_call', 'workflow_dispatch'})

    def test_required_checks_always_run_without_workflow_path_filters(self):
        ci = workflow('roadmap-leases.yml')
        for event in ci['on'].values():
            if isinstance(event, dict):
                self.assertFalse({'paths', 'paths-ignore'} & set(event))
        jobs = ci['jobs']
        self.assertNotIn('if', jobs['plan'])
        self.assertEqual(jobs['plan']['name'], 'Roadmap lease contract')
        self.assertEqual(jobs['required']['name'], 'CI required')
        self.assertEqual(jobs['required']['if'], 'always()')
        self.assertEqual(set(jobs['required']['needs']), {'plan', *SUITES})
        final = jobs['required']['steps'][-1]
        self.assertEqual(final['env']['CI_NEEDS'], '${{ toJSON(needs) }}')
        self.assertEqual(final['run'], 'python3 tools/ci/check_result.py')

    def test_routes_and_reusable_calls_are_all_wired_and_conditioned(self):
        jobs = workflow('roadmap-leases.yml')['jobs']
        self.assertEqual(set(jobs), {'plan', 'required', *SUITES})
        self.assertEqual(set(jobs['plan']['outputs']), {'plan', *SUITES})
        for suite in SUITES:
            job = jobs[suite]
            self.assertEqual(job['needs'], 'plan')
            self.assertEqual(job['if'], f"needs.plan.outputs.{suite} == 'true'")
            self.assertEqual(jobs['plan']['outputs'][suite], '${{ steps.route.outputs.' + suite + ' }}')
            called = ROOT / job['uses'].removeprefix('./')
            self.assertTrue(called.is_file())
            self.assertIn('workflow_call', workflow(called.name)['on'])

    def test_two_short_linux_jobs_only_when_every_suite_is_skipped(self):
        jobs = workflow('roadmap-leases.yml')['jobs']
        hosted = {name: job for name, job in jobs.items() if 'runs-on' in job}
        self.assertEqual(set(hosted), {'plan', 'required'})
        self.assertTrue(all(j['runs-on'] == 'ubuntu-latest' for j in hosted.values()))
        self.assertTrue(all(int(j['timeout-minutes']) <= 5 for j in hosted.values()))

    def test_pr_updates_cancel_but_main_pushes_have_distinct_groups(self):
        ci = workflow('roadmap-leases.yml')
        self.assertEqual(ci['concurrency']['cancel-in-progress'], "${{ github.event_name == 'pull_request' }}")
        self.assertIn('github.event_name', ci['concurrency']['group'])
        self.assertIn('github.event.pull_request.number || github.run_id', ci['concurrency']['group'])
        for path in (ROOT / '.github/workflows').glob('*.yml'):
            if path.name != 'roadmap-leases.yml':
                self.assertNotIn('concurrency', workflow(path.name))

    def test_read_only_permissions_and_full_history_diff(self):
        for path in (ROOT / '.github/workflows').glob('*.yml'):
            self.assertEqual(workflow(path.name)['permissions'], {'contents': 'read'})
        steps = workflow('roadmap-leases.yml')['jobs']['plan']['steps']
        checkout = next(s for s in steps if s.get('uses', '').startswith('actions/checkout@'))
        self.assertEqual(checkout['with']['fetch-depth'], '0')
        self.assertTrue(any(s.get('id') == 'route' and s['run'] == 'python3 tools/ci/route.py' for s in steps))
        self.assertTrue(any('discover -s tests/ci' in s.get('run', '') for s in steps))
        self.assertTrue(any('discover -s tests/roadmap' in s.get('run', '') for s in steps))

    def test_full_platform_suite_contents_are_retained(self):
        shader_job = workflow('shader-portability.yml')['jobs']['compile']
        self.assertEqual(len(shader_job['strategy']['matrix']['include']), 3)
        validation = next(
            step for step in shader_job['steps']
            if step.get('name') == 'Validate reflection and host contracts')
        self.assertIn('slang-d3d12-realtime-tests', validation['run'])
        self.assertEqual(set(workflow('sdk-install.yml')['jobs']), {'sdk', 'sdk-macos', 'sdk-windows'})
        self.assertEqual(set(workflow('mobile-gallery.yml')['jobs']), {'ipad', 'android', 'android-ui'})
        self.assertEqual(len(workflow('gallery-frame-timing.yml')['jobs']['portable-timing']['strategy']['matrix']['os']), 3)
        self.assertIn('reproduce', workflow('reference-artifacts.yml')['jobs'])
        self.assertIn('loader', workflow('windows-loader.yml')['jobs'])


if __name__ == '__main__':
    unittest.main()
