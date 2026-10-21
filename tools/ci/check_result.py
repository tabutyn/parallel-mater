# SPDX-License-Identifier: MIT
"""Fail the stable required check unless every selected suite succeeded."""
import json
import os
import sys

from route import SUITES


def check(needs):
    if not isinstance(needs, dict) or set(needs) != {'plan', *SUITES}:
        raise ValueError('missing or unexpected CI jobs')
    planner = needs['plan']
    if planner['result'] != 'success':
        raise ValueError('routing/contract checks did not succeed')
    plan = json.loads(planner['outputs']['plan'])
    if (not isinstance(plan, dict) or set(plan) != set(SUITES)
            or any(type(value) is not bool for value in plan.values())):
        raise ValueError('invalid routing outputs')
    for suite, required in plan.items():
        expected = 'success' if required else 'skipped'
        if needs[suite]['result'] != expected:
            raise ValueError(f'{suite}: expected {expected}, got {needs[suite]["result"]}')


def main():
    try:
        check(json.loads(os.environ['CI_NEEDS']))
    except (KeyError, TypeError, ValueError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print('PASS: contracts and all selected CI suites succeeded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
