# SPDX-License-Identifier: MIT
"""Conservative CI routing. Unknown paths or unavailable diffs select everything."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

SUITES = ('shaders', 'sdk', 'reference', 'timing', 'windows', 'mobile')
PROSE_FILES = {'README.md', 'CONTRIBUTING.md', 'AGENTS.md', 'LICENSE',
               '.github/pull_request_template.md'}


def classify(paths, full=False):
    selected = set(SUITES) if full else set()
    for path in paths:
        # These are generated/tested artifacts, NOT ordinary documentation.
        if path.startswith(('docs/reference/', 'tools/reference/')) or path == 'tests/release/test_reference_artifacts.py':
            selected.add('reference')
        elif (path in PROSE_FILES or path == 'tools/roadmap.py'
              or path.startswith(('docs/roadmap/', 'docs/history/', 'tests/roadmap/'))
              or (path.startswith('docs/') and path.endswith('.md'))):
            pass  # Always-run contracts still test the roadmap and routing.
        else:
            # Shared code/build inputs, CI configuration and every unknown path.
            # Narrow this only with an audited dependency map and regression test.
            selected.update(SUITES)
    return {suite: suite in selected for suite in SUITES}


def commit(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{40,64}', value) or not value.strip('0'):
        raise ValueError('missing or invalid event commit')
    return value


def changed_paths(base, head, *, merge_base=False, cwd=None):
    base, head = commit(base), commit(head)
    if merge_base:
        base = subprocess.check_output(['git', 'merge-base', base, head], cwd=cwd,
                                       stderr=subprocess.PIPE, text=True).strip()
        commit(base)
    # Treat renames as delete+add: both paths affect routing. NUL delimiters
    # preserve spaces, newlines and non-ASCII paths; no API/file-count truncation.
    result = subprocess.check_output(
        ['git', 'diff', '--name-only', '--no-renames', '-z', base, head, '--'],
        cwd=cwd, stderr=subprocess.PIPE)
    return [p.decode('utf-8', errors='surrogateescape') for p in result.split(b'\0') if p]


def plan_for_event(name, event, *, cwd=None):
    try:
        if name == 'pull_request':
            pr = event['pull_request']
            paths = changed_paths(pr['base']['sha'], pr['head']['sha'], merge_base=True, cwd=cwd)
        elif name == 'push':
            paths = changed_paths(event['before'], event['after'], cwd=cwd)
        elif name == 'merge_group':
            group = event['merge_group']
            paths = changed_paths(group['base_sha'], group['head_sha'], cwd=cwd)
        else:
            return classify([], full=True), f'{name}: full verification'
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        return classify([], full=True), f'{name}: diff unavailable; full verification'
    return classify(paths), f'{name}: classified {len(paths)} changed paths'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-name', default=os.environ.get('GITHUB_EVENT_NAME', 'unknown'))
    parser.add_argument('--event-path', default=os.environ.get('GITHUB_EVENT_PATH'))
    parser.add_argument('--output', default=os.environ.get('GITHUB_OUTPUT'))
    parser.add_argument('--summary', default=os.environ.get('GITHUB_STEP_SUMMARY'))
    args = parser.parse_args()
    try:
        event = json.loads(Path(args.event_path).read_text()) if args.event_path else {}
        plan, reason = plan_for_event(args.event_name, event)
    except (OSError, ValueError):
        plan, reason = classify([], full=True), 'unreadable event; full verification'
    print(json.dumps({'reason': reason, 'suites': plan}, sort_keys=True))
    if args.output:
        with open(args.output, 'a', encoding='utf-8') as stream:
            stream.write('plan=' + json.dumps(plan, sort_keys=True) + '\n')
            for suite, enabled in plan.items():
                stream.write(f'{suite}={str(enabled).lower()}\n')
    if args.summary:
        with open(args.summary, 'a', encoding='utf-8') as stream:
            stream.write('## CI selection\n\n' + reason + '\n\n')
            stream.write('Roadmap and CI contract tests: always run.\n\n')
            for suite, enabled in plan.items():
                stream.write(f'- {suite}: {"selected" if enabled else "not applicable"}\n')
            stream.write('\nNot applicable is not platform execution/qualification evidence.\n')


if __name__ == '__main__':
    main()
