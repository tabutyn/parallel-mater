#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cooperative, PR-serialized roadmap leases. Git main is the ownership record."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = Path('docs/roadmap/tasks')
ROADMAP = Path('docs/ROADMAP.md')
BEGIN, END = '<!-- BEGIN TASK INDEX -->', '<!-- END TASK INDEX -->'
FIELDS = {'id', 'title', 'status', 'dependencies', 'locks', 'requires', 'scope',
          'acceptance', 'evidence', 'lease', 'history'}
LEASE_FIELDS = {'token', 'owner', 'machine', 'branch', 'claimed_at', 'heartbeat_at', 'expires_at'}
GROUPS = {'F1': 'Resource lifetime and capacity', 'F2': 'Interaction graph and system coupling',
          'F3': 'Specification, ABI and validation', 'F4': 'Common resources and engine ownership',
          'F5': 'Independent conformance and quality', 'F6': 'SDK releases and repository ownership',
          'F7': 'Shared Gallery application',
          'P1': 'Deferred domain and interaction coverage', 'P2': 'Deferred engine integration',
          'P3': 'Deferred measured optimization', 'P4': 'Deferred vendor and standards adoption'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', value),
            f'UTC timestamp must be YYYY-MM-DDTHH:MM:SSZ: {value!r}')
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def strings(value, label, nonempty=False):
    require(isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value),
            f'{label}: expected a list of nonempty strings')
    require(not nonempty or value, f'{label}: must not be empty')
    require(len(value) == len(set(value)), f'{label}: duplicate entries')


def validate_lease(lease):
    require(isinstance(lease, dict) and set(lease) == LEASE_FIELDS, 'invalid lease fields')
    for key in LEASE_FIELDS:
        require(isinstance(lease[key], str) and lease[key].strip(), f'lease {key} is empty')
    require(str(uuid.UUID(lease['token'])) == lease['token'], 'lease token must be a canonical UUID')
    require(lease['branch'].startswith('work/') and not any(c.isspace() for c in lease['branch']),
            'implementation branch must start with work/ and contain no whitespace')
    claimed, heartbeat, expires = (timestamp(lease[key]) for key in
                                   ('claimed_at', 'heartbeat_at', 'expires_at'))
    require(claimed <= heartbeat < expires <= heartbeat + timedelta(hours=24),
            'lease must expire within 24 hours of its heartbeat')


def validate(tasks):
    tokens, held = set(), {}
    for id, task in tasks.items():
        require(set(task) == FIELDS, f'{id}: unexpected or missing task fields')
        require(task['id'] == id and re.fullmatch(r'(?:F[1-7]|P[1-4])-\d\d', id), f'invalid ID: {id}')
        require(task['status'] in ('open', 'leased', 'done', 'deferred'), f'{id}: invalid status')
        for key in ('title', 'scope'):
            require(isinstance(task[key], str) and task[key].strip(), f'{id}: missing {key}')
        for key in ('dependencies', 'locks', 'requires', 'acceptance', 'evidence'):
            strings(task[key], f'{id}.{key}', key in ('requires', 'acceptance'))
        require(all(dep in tasks and dep != id for dep in task['dependencies']), f'{id}: bad dependency')
        require(isinstance(task['history'], list), f'{id}: history must be a list')
        for event in task['history']:
            require(set(event) == {'lease', 'outcome', 'at', 'reason', 'evidence'}, f'{id}: invalid history')
            validate_lease(event['lease'])
            require(event['outcome'] in ('released', 'reclaimed', 'completed'), f'{id}: invalid outcome')
            require(timestamp(event['at']) >= timestamp(event['lease']['claimed_at']), f'{id}: invalid history time')
            require(isinstance(event['reason'], str) and event['reason'].strip(), f'{id}: history needs reason')
            strings(event['evidence'], f'{id}.history.evidence')
            require(event['lease']['token'] not in tokens, f'{id}: reused lease token')
            tokens.add(event['lease']['token'])
        if task['status'] == 'leased':
            validate_lease(task['lease'])
            require(task['lease']['token'] not in tokens, f'{id}: reused lease token')
            tokens.add(task['lease']['token'])
            require(all(tasks[d]['status'] == 'done' for d in task['dependencies']),
                    f'{id}: unfinished dependency')
            for lock in task['locks']:
                require(lock not in held, f'{id}: lock {lock} already held by {held.get(lock)}')
                held[lock] = id  # Expired leases still reserve locks until a PR retires them.
        else:
            require(task['lease'] is None, f'{id}: only leased tasks can have a current lease')
        if task['status'] == 'done':
            require(task['evidence'], f'{id}: completed task needs evidence')
            require(all(tasks[d]['status'] == 'done' for d in task['dependencies']),
                    f'{id}: completed with unfinished dependencies')
    def visit(id, active, seen):
        require(id not in active, f'dependency cycle at {id}')
        if id in seen:
            return
        for dependency in tasks[id]['dependencies']:
            visit(dependency, active | {id}, seen)
        seen.add(id)
    seen = set()
    for id in tasks:
        visit(id, set(), seen)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def load(root, revision=None):
    if revision:
        paths = git(root, 'ls-tree', '-r', '--name-only', revision, '--', TASK_DIR.as_posix()).splitlines()
        result = {Path(p).stem: json.loads(git(root, 'show', f'{revision}:{p}'))
                  for p in paths if p.endswith('.json')}
    else:
        result = {p.stem: json.loads(p.read_text(encoding='utf-8'))
                  for p in sorted((root / TASK_DIR).glob('*.json'))}
    validate(result)
    return result


def available(task, tasks):
    if task['status'] != 'open':
        return task['status']
    if any(tasks[d]['status'] != 'done' for d in task['dependencies']):
        return 'waiting on dependencies'
    held = {lock for other in tasks.values() if other['status'] == 'leased' for lock in other['locks']}
    return 'waiting on lock' if held.intersection(task['locks']) else 'ready'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def index(tasks):
    rows = ['<!-- Generated by tools/roadmap.py render. Edit task cards, not this index. -->', '']
    for group, title in GROUPS.items():
        selected = [t for id, t in sorted(tasks.items()) if id.startswith(group + '-')]
        rows += [f'### {group}. {title}', '',
                 '| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |',
                 '|---|---|---|---|---|']
        for task in selected:
            lease = task['lease']
            owner = (f"{lease['owner']} / {lease['machine']} / {lease['expires_at']}" if lease else '—')
            rows.append(f"| [{task['id']}](roadmap/tasks/{task['id']}.json) | {cell(task['title'])} | "
                        f"{available(task, tasks)} | {', '.join(task['dependencies']) or '—'} | {cell(owner)} |")
        rows += ['']
    return '\n'.join(rows)


def rendered(root, tasks):
    text = (root / ROADMAP).read_text(encoding='utf-8')
    require(text.count(BEGIN) == text.count(END) == 1, 'roadmap must have one task index block')
    before, tail = text.split(BEGIN)
    _, after = tail.split(END)
    return before + BEGIN + '\n' + index(tasks) + END + after


def coordination_path(path):
    return (path in ('docs/ROADMAP.md', 'CONTRIBUTING.md', 'AGENTS.md', 'tools/roadmap.py',
                     '.github/workflows/roadmap-leases.yml', '.github/pull_request_template.md') or
            path.startswith(('docs/roadmap/', 'tests/roadmap/', 'docs/history/ROADMAP-')))


def transitions(base, tasks, now, changed_paths=(), branch=None):
    """Validate against current main, not a stale merge base."""
    require(set(base) <= set(tasks), 'do not delete/reuse task IDs; retain history')
    changes = []
    for id, current in tasks.items():
        if id not in base:
            require(current['lease'] is None, f'{id}: add a task before claiming it')
            require(not base or current['status'] in ('open', 'deferred'),
                    f'{id}: new tasks start open/deferred, not completed')
            continue
        previous = base[id]
        if current == previous:
            continue
        prior, lease = previous['lease'], current['lease']
        operational = ('status', 'lease', 'history', 'evidence')
        definition_changed = any(current[k] != previous[k] for k in FIELDS - set(operational))
        if prior:
            require(not definition_changed, f'{id}: freeze scope/dependencies/locks during lease; release before replanning')
        if lease and (not prior or lease['token'] != prior['token']):
            require(not definition_changed, f'{id}: define scope in a planning PR before claiming')
            require(previous['status'] in ('open', 'leased'), f'{id}: task must be open or expired leased')
            require(all(base[d]['status'] == 'done' for d in current['dependencies']), f'{id}: dependencies must already be merged done')
            require(lease['heartbeat_at'] == lease['claimed_at'], f'{id}: new lease heartbeat must match its claim')
            require(timestamp(lease['claimed_at']) <= now + timedelta(minutes=5), f'{id}: claim is in the future')
            require(timestamp(lease['expires_at']) > now, f'{id}: new lease already expired')
            changes.append(('reclaim' if prior else 'claim', id))
        elif prior and lease:
            for key in ('token', 'owner', 'machine', 'branch', 'claimed_at'):
                require(lease[key] == prior[key], f'{id}: renew cannot change {key}')
            require(timestamp(prior['expires_at']) > now, f'{id}: expired lease must be reclaimed with a new token')
            require(timestamp(lease['heartbeat_at']) >= timestamp(prior['heartbeat_at']), f'{id}: heartbeat moved backwards')
            require(timestamp(lease['heartbeat_at']) <= now + timedelta(minutes=5), f'{id}: heartbeat in future')
            require(timestamp(lease['expires_at']) >= timestamp(prior['expires_at']), f'{id}: renewal shortens lease')
            changes.append(('renew', id))
        elif prior:
            require(current['status'] in ('open', 'done'), f'{id}: release to open or complete to done')
            if current['status'] == 'done':
                require(timestamp(prior['expires_at']) > now, f'{id}: cannot complete expired lease')
                if branch:
                    require(branch == prior['branch'], f'{id}: completion must use {prior["branch"]}')
            changes.append(('complete' if current['status'] == 'done' else 'release', id))
        else:
            require(current['status'] != 'done' or previous['status'] == 'done', f'{id}: completion needs a merged lease')
            require(previous['status'] != 'done' or current['status'] == 'done', f'{id}: create a new follow-up task instead of reopening completed history')
        retiring = prior and (not lease or prior['token'] != lease['token'])
        expected_history = previous['history']
        if retiring:
            require(len(current['history']) == len(expected_history) + 1 and
                    current['history'][:-1] == expected_history, f'{id}: append exactly one retirement event')
            event = current['history'][-1]
            outcome = 'reclaimed' if lease else 'completed' if current['status'] == 'done' else 'released'
            require(event['lease'] == prior and event['outcome'] == outcome, f'{id}: stale lease token or incorrect retirement')
            require(timestamp(event['at']) <= now + timedelta(minutes=5), f'{id}: retirement in future')
            if outcome == 'reclaimed':
                require(timestamp(prior['expires_at']) <= now and timestamp(prior['expires_at']) <= timestamp(event['at']),
                        f'{id}: cannot take another live lease; owner must release it first')
            if outcome == 'completed':
                require(current['evidence'] and event['evidence'] == current['evidence'], f'{id}: completion must retain matching evidence')
        else:
            require(current['history'] == expected_history, f'{id}: immutable lease history')
        if current['status'] != 'done':
            require(current['evidence'] == previous['evidence'], f'{id}: evidence is recorded by completion')
    require(len(changes) <= 1, 'one lease operation/task per PR')
    if changes and changes[0][0] == 'complete':
        require(all(tasks[other] == base[other] for other in base if other != changes[0][1]),
                'completion PR cannot redefine another existing task; use a planning PR')
    if changes and changes[0][0] != 'complete':
        _, id = changes[0]
        # Git reports repository paths with '/', including on Windows.
        allowed = {ROADMAP.as_posix(), (TASK_DIR / f'{id}.json').as_posix()}
        require(set(changed_paths) <= allowed, 'claim/renew/reclaim/release PR must contain only its task card and generated roadmap')
        require(all(tasks[other] == base[other] for other in base if other != id), 'lease PR may change only its one task')
        require(set(tasks) == set(base), 'lease PR cannot add tasks')
    if base and any(not coordination_path(path) for path in changed_paths):
        require(changes and changes[0][0] == 'complete', 'implementation PR must complete one currently leased task')
    return changes


def update(tasks, id, action, now, args):
    require(id in tasks, f'unknown task: {id}')
    task = tasks[id]
    previous = json.loads(json.dumps(task))
    prior = task['lease']
    if action in ('claim', 'reclaim'):
        require((action == 'claim' and available(task, tasks) == 'ready') or
                (action == 'reclaim' and prior and timestamp(prior['expires_at']) <= now),
                f'{id}: not available for {action}')
        require(all((args.owner, args.machine, args.branch)), 'claim/reclaim needs --owner, --machine and --branch')
        task['lease'] = {'token': str(uuid.uuid4()), 'owner': args.owner, 'machine': args.machine,
                         'branch': args.branch, 'claimed_at': stamp(now), 'heartbeat_at': stamp(now),
                         'expires_at': stamp(now + timedelta(hours=args.hours))}
        task['status'] = 'leased'
    else:
        require(prior and args.token == prior['token'], f'{id}: --token must match current lease')
        if action == 'renew':
            require(timestamp(prior['expires_at']) > now, 'expired lease needs reclaim')
            task['lease'] = dict(prior, heartbeat_at=stamp(now),
                                 expires_at=stamp(max(timestamp(prior['expires_at']), now + timedelta(hours=args.hours))))
        else:
            task['lease'] = None
            task['status'] = 'done' if action == 'complete' else 'open'
    if action in ('release', 'reclaim', 'complete'):
        require(args.reason, f'{action} requires --reason')
        evidence = list(args.evidence or []) if action == 'complete' else []
        if action == 'complete':
            require(timestamp(prior['expires_at']) > now, 'expired lease cannot complete')
            require(evidence, 'complete needs reviewable --evidence (repeatable)')
            task['evidence'] = evidence
        task['history'].append({'lease': prior, 'outcome': {'release':'released','reclaim':'reclaimed','complete':'completed'}[action],
                                'at': stamp(now), 'reason': args.reason, 'evidence': evidence})
    try:
        validate(tasks)
    except Exception:
        tasks[id] = previous
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'render', 'list'):
        sub = commands.add_parser(name)
        if name == 'check':
            sub.add_argument('--base', help='Fetched current main revision; validates PR transitions')
            sub.add_argument('--branch', help='Implementation PR head branch (CI)')
        if name == 'list':
            sub.add_argument('--ready', action='store_true')
    for name in ('claim', 'renew', 'release', 'reclaim', 'complete'):
        sub = commands.add_parser(name)
        sub.add_argument('id')
        sub.add_argument('--owner')
        sub.add_argument('--machine')
        sub.add_argument('--branch')
        sub.add_argument('--token')
        sub.add_argument('--hours', type=int, default=24, choices=range(1,25))
        sub.add_argument('--reason')
        sub.add_argument('--evidence', action='append')
    args = parser.parse_args()
    root, now = args.root.resolve(), datetime.now(timezone.utc).replace(microsecond=0)
    try:
        tasks = load(root)
        require(tasks, 'no task cards found')
        if args.command == 'list':
            for id, task in sorted(tasks.items()):
                state = available(task, tasks)
                if args.ready and state != 'ready':
                    continue
                if task['lease'] and timestamp(task['lease']['expires_at']) <= now:
                    state = 'expired; explicit reclaim required'
                print(f"{id:6} {state:27} {task['title']} ({'; '.join(task['requires'])})")
        elif args.command == 'check':
            require((root / ROADMAP).read_text(encoding='utf-8') == rendered(root,tasks), 'stale roadmap index: run tools/roadmap.py render')
            if args.base:
                base = load(root,args.base)
                if base:
                    require(subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',args.base,'HEAD'],check=False).returncode == 0,
                            'update this branch with current main before checking/merging')
                changed = git(root,'diff','--name-only',args.base,'--').splitlines()
                transitions(base,tasks,now,changed,args.branch)
            print(f'PASS: {len(tasks)} task cards, dependencies, locks, history and generated roadmap')
        elif args.command == 'render':
            (root / ROADMAP).write_text(rendered(root,tasks),encoding='utf-8')
        else:
            update(tasks,args.id,args.command,now,args)
            content = rendered(root,tasks)
            (root / TASK_DIR / f'{args.id}.json').write_text(json.dumps(tasks[args.id],indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            (root / ROADMAP).write_text(content,encoding='utf-8')
            print(f"Prepared {args.command} for {args.id}; push a PR. Ownership changes only on merged main.")
            if tasks[args.id]['lease']:
                print('Lease token:',tasks[args.id]['lease']['token'])
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
