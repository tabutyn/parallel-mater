# SPDX-License-Identifier: MIT
import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('roadmap', ROOT / 'tools/roadmap.py')
roadmap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roadmap)
NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


def task(id='F1-01', dependencies=(), locks=()):
    return dict(id=id, title='Bounded task', status='open', dependencies=list(dependencies),
                locks=list(locks), requires=['host'], scope='One reviewed change',
                acceptance=['Independent public contract test'], evidence=[], lease=None, history=[])


def options(**values):
    defaults = dict(owner='agent-a', machine='linux-a', branch='work/F1-01/agent-a',
                    hours=24, token=None, reason='Reviewable reason', evidence=['PASS: tracked test report'])
    return argparse.Namespace(**(defaults | values))


def claim(tasks, id='F1-01', now=NOW, **values):
    roadmap.update(tasks, id, 'claim', now, options(**values))
    return tasks[id]['lease']['token']


def paths(id='F1-01'):
    return ['docs/ROADMAP.md', f'docs/roadmap/tasks/{id}.json']


class LeaseTests(unittest.TestCase):
    def test_catalog_is_valid_and_index_current(self):
        tasks = roadmap.load(ROOT)
        self.assertEqual((ROOT / roadmap.ROADMAP).read_text(), roadmap.rendered(ROOT, tasks))
        self.assertGreaterEqual(len(tasks), 60)

    def test_registered_f8_ids_validate_but_unknown_groups_do_not(self):
        roadmap.validate({'F8-00': task('F8-00')})
        for id in ('F9-00', 'P5-00', 'F8-1', 'F08-00'):
            with self.assertRaisesRegex(ValueError, 'invalid ID'):
                roadmap.validate({id: task(id)})

    def test_only_entirely_completed_categories_are_collapsed(self):
        done = task() | {'status': 'done', 'evidence': ['Historical evidence']}
        for status in ('open', 'leased', 'deferred'):
            tasks = {'F1-01': done, 'F8-01': task('F8-01')}
            if status == 'leased':
                claim(tasks, 'F8-01')
            else:
                tasks['F8-01']['status'] = status
            roadmap.validate(tasks)
            before = copy.deepcopy(tasks)
            text = roadmap.index(tasks)
            active, archived = text.split('### Completed categories')
            self.assertIn('### F8.', active)
            self.assertNotIn('### F1.', active)
            self.assertNotIn('### F2.', text)  # Empty groups are not completed.
            self.assertIn('<summary>F1.', archived)
            self.assertIn('roadmap/tasks/F1-01.json', archived)
            self.assertNotIn('<summary>F8.', archived)
            self.assertEqual(before, tasks)  # Rendering never rewrites history.

    def test_new_followup_makes_completed_category_visible_again(self):
        tasks = {'F8-01': task('F8-01') | {'status': 'done', 'evidence': ['PASS']}}
        self.assertIn('<summary>F8.', roadmap.index(tasks))
        tasks['F8-02'] = task('F8-02', dependencies=['F8-01'])
        text = roadmap.index(tasks)
        self.assertIn('### F8.', text)
        self.assertNotIn('<summary>F8.', text)
        self.assertIn('roadmap/tasks/F8-01.json', text)
        self.assertIn('roadmap/tasks/F8-02.json', text)

    def test_new_claim_is_metadata_only_and_needs_merged_dependencies(self):
        base = {'F1-01': task()}
        candidate = copy.deepcopy(base)
        claim(candidate)
        self.assertEqual(roadmap.transitions(base, candidate, NOW, paths()), [('claim', 'F1-01')])
        with self.assertRaisesRegex(ValueError, 'only its task card'):
            roadmap.transitions(base, candidate, NOW, paths() + ['src/physics.cpp'])
        waiting = {'F1-01': task(), 'F1-02': task('F1-02', ['F1-01'])}
        with self.assertRaisesRegex(ValueError, 'not available'):
            claim(waiting, 'F1-02')

    def test_competing_claim_on_another_task_has_exclusive_lock(self):
        tasks = {'F1-01': task(locks=['memory']), 'F1-02': task('F1-02', locks=['memory'])}
        claim(tasks)
        self.assertEqual(roadmap.available(tasks['F1-02'],tasks), 'waiting on lock')
        with self.assertRaisesRegex(ValueError, 'not available'):
            claim(tasks, 'F1-02')
        # Passing time does not silently release ownership or shared locks.
        with self.assertRaisesRegex(ValueError, 'not available'):
            claim(tasks, 'F1-02', NOW + timedelta(days=2))

    def test_two_branches_claiming_same_task_cannot_both_land(self):
        base = {'F1-01': task()}
        first, second = copy.deepcopy(base), copy.deepcopy(base)
        claim(first)
        claim(second, owner='agent-b', machine='mac-b', branch='work/F1-01/agent-b')
        with self.assertRaises(ValueError):
            roadmap.transitions(first, second, NOW, paths())

    def test_renew_preserves_identity_and_expired_renew_fails(self):
        base = {'F1-01': task()}
        token = claim(base)
        candidate = copy.deepcopy(base)
        roadmap.update(candidate,'F1-01','renew',NOW + timedelta(hours=1),options(token=token))
        roadmap.transitions(base,candidate,NOW + timedelta(hours=1),paths())
        self.assertEqual(candidate['F1-01']['lease']['token'],token)
        changed = copy.deepcopy(candidate)
        changed['F1-01']['lease']['owner']='agent-b'
        with self.assertRaisesRegex(ValueError, 'cannot change owner'):
            roadmap.transitions(base,changed,NOW + timedelta(hours=1),paths())
        with self.assertRaisesRegex(ValueError, 'expired'):
            roadmap.update(candidate,'F1-01','renew',NOW + timedelta(days=2),options(token=token))

    def test_early_reclaim_fails_and_expired_reclaim_retires_token(self):
        base={'F1-01':task()}
        old_token=claim(base)
        with self.assertRaisesRegex(ValueError, 'not available'):
            roadmap.update(copy.deepcopy(base),'F1-01','reclaim',NOW,options())
        candidate=copy.deepcopy(base)
        later=NOW+timedelta(days=2)
        roadmap.update(candidate,'F1-01','reclaim',later,options(owner='agent-b',machine='mac-b',branch='work/F1-01/agent-b'))
        roadmap.transitions(base,candidate,later,paths())
        self.assertNotEqual(old_token,candidate['F1-01']['lease']['token'])
        self.assertEqual(candidate['F1-01']['history'][0]['lease']['token'],old_token)
        self.assertEqual(candidate['F1-01']['history'][0]['outcome'],'reclaimed')

    def test_old_completion_rejected_after_reclaim(self):
        original={'F1-01':task()}
        token=claim(original)
        stale=copy.deepcopy(original)
        roadmap.update(stale,'F1-01','complete',NOW,options(token=token))
        main=copy.deepcopy(original)
        later=NOW+timedelta(days=2)
        roadmap.update(main,'F1-01','reclaim',later,options(owner='agent-b',branch='work/F1-01/agent-b'))
        # Even omitting branch identity, history/token comparison rejects stale work.
        with self.assertRaises(ValueError):
            roadmap.transitions(main,stale,later,paths()+['src/physics.cpp'])

    def test_complete_requires_token_evidence_and_work_branch(self):
        base={'F1-01':task()}
        token=claim(base)
        with self.assertRaisesRegex(ValueError, 'token'):
            roadmap.update(copy.deepcopy(base),'F1-01','complete',NOW,options(token='wrong'))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            roadmap.update(copy.deepcopy(base),'F1-01','complete',NOW,options(token=token,evidence=[]))
        candidate=copy.deepcopy(base)
        roadmap.update(candidate,'F1-01','complete',NOW,options(token=token))
        roadmap.transitions(base,candidate,NOW,paths()+['src/physics.cpp'],branch='work/F1-01/agent-a')
        with self.assertRaisesRegex(ValueError, 'completion must use'):
            roadmap.transitions(base,candidate,NOW,paths(),branch='work/F1-01/agent-b')

    def test_expired_completion_fails(self):
        base={'F1-01':task()}
        token=claim(base)
        with self.assertRaisesRegex(ValueError, 'expired'):
            roadmap.update(copy.deepcopy(base),'F1-01','complete',NOW+timedelta(days=2),options(token=token))

    def test_release_retains_history_and_frees_lock(self):
        base={'F1-01':task(locks=['memory']), 'F1-02':task('F1-02',locks=['memory'])}
        token=claim(base)
        candidate=copy.deepcopy(base)
        roadmap.update(candidate,'F1-01','release',NOW,options(token=token))
        roadmap.transitions(base,candidate,NOW,paths())
        self.assertEqual(roadmap.available(candidate['F1-02'],candidate),'ready')
        self.assertEqual(candidate['F1-01']['history'][0]['lease']['token'],token)
        edited=copy.deepcopy(candidate)
        edited['F1-01']['history']=[]
        with self.assertRaisesRegex(ValueError,'history'):
            roadmap.transitions(candidate,edited,NOW,paths())

    def test_cannot_claim_and_change_scope_or_add_task(self):
        base={'F1-01':task()}
        candidate=copy.deepcopy(base)
        claim(candidate)
        candidate['F1-01']['scope']='Unreviewed extra work'
        with self.assertRaisesRegex(ValueError,'planning PR'):
            roadmap.transitions(base,candidate,NOW,paths())
        candidate=copy.deepcopy(base)
        claim(candidate)
        candidate['F1-02']=task('F1-02')
        with self.assertRaisesRegex(ValueError,'cannot add tasks'):
            roadmap.transitions(base,candidate,NOW,paths())

    def test_live_scope_and_locks_are_frozen(self):
        base={'F1-01':task()}
        claim(base)
        candidate=copy.deepcopy(base)
        candidate['F1-01']['locks']=['new-area']
        with self.assertRaisesRegex(ValueError,'freeze scope'):
            roadmap.transitions(base,candidate,NOW,paths())

    def test_implementation_requires_completion_and_only_one_task(self):
        base={'F1-01':task(), 'F1-02':task('F1-02')}
        with self.assertRaisesRegex(ValueError,'implementation PR'):
            roadmap.transitions(base,base,NOW,['src/physics.cpp'])
        candidate=copy.deepcopy(base)
        claim(candidate)
        claim(candidate,'F1-02')
        with self.assertRaisesRegex(ValueError,'one lease operation'):
            roadmap.transitions(base,candidate,NOW,paths()+paths('F1-02'))

    def test_cycles_unknown_dependencies_and_missing_evidence_fail(self):
        for tasks in ({'F1-01':task(dependencies=['F1-02'])},
                      {'F1-01':task(dependencies=['F1-02']), 'F1-02':task('F1-02',['F1-01'])},
                      {'F1-01':task() | {'status':'done'}}):
            with self.assertRaises(ValueError):
                roadmap.validate(tasks)

    def test_new_tasks_need_planning_before_claim(self):
        base={'F1-01':task()}
        candidate=copy.deepcopy(base)
        candidate['F1-02']=task('F1-02')
        claim(candidate,'F1-02')
        with self.assertRaisesRegex(ValueError,'before claiming'):
            roadmap.transitions(base,candidate,NOW,paths('F1-02'))

    def test_deferred_tasks_and_completed_ids_cannot_be_reused(self):
        base={'F1-01':task() | {'status':'deferred'}}
        with self.assertRaisesRegex(ValueError,'not available'):
            claim(copy.deepcopy(base))
        base['F1-01'].update(status='done',evidence=['Historical commit'])
        candidate=copy.deepcopy(base)
        candidate['F1-01']['status']='open'
        with self.assertRaisesRegex(ValueError,'follow-up'):
            roadmap.transitions(base,candidate,NOW,paths())

    def test_heartbeat_cannot_create_future_week_long_claim(self):
        base={'F1-01':task()}
        candidate=copy.deepcopy(base)
        claim(candidate)
        candidate['F1-01']['lease']['heartbeat_at']=roadmap.stamp(NOW+timedelta(days=7))
        candidate['F1-01']['lease']['expires_at']=roadmap.stamp(NOW+timedelta(days=8))
        with self.assertRaisesRegex(ValueError,'heartbeat must match'):
            roadmap.transitions(base,candidate,NOW,paths())

    def test_cli_rejects_stale_main_even_for_different_task_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/roadmap.TASK_DIR).mkdir(parents=True)
            (root/roadmap.ROADMAP).write_text('# Queue\n'+roadmap.BEGIN+'\n'+roadmap.END+'\n')
            for id in ('F1-01','F1-02'):
                (root/roadmap.TASK_DIR/(id+'.json')).write_text(json.dumps(task(id)))
            (root/roadmap.ROADMAP).write_text(roadmap.rendered(root,roadmap.load(root)))
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
            def commit(message):
                git('add','.')
                git('-c','user.name=Lease Test','-c','user.email=lease@example.invalid',
                    '-c','commit.gpgsign=false','commit','-m',message)
            git('init','-b','main')
            commit('base')
            git('switch','-c','claim-a')
            command=[sys.executable,str(ROOT/'tools/roadmap.py'),'--root',str(root)]
            subprocess.run(command+['claim','F1-01','--owner','a','--machine','linux',
                '--branch','work/F1-01/a'],check=True,capture_output=True)
            commit('claim a')
            git('switch','main')
            (root/'coordination-note.txt').write_text('main advanced independently')
            commit('main advanced')
            git('switch','claim-a')
            result=subprocess.run(command+['check','--base','main'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('update this branch with current main',result.stderr)
            git('-c','user.name=Lease Test','-c','user.email=lease@example.invalid',
                '-c','commit.gpgsign=false','merge','main','--no-edit')
            result=subprocess.run(command+['check','--base','main'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_cli_claim_regenerates_visible_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/roadmap.TASK_DIR).mkdir(parents=True)
            (root/roadmap.ROADMAP).write_text('# Queue\n'+roadmap.BEGIN+'\n'+roadmap.END+'\n')
            (root/roadmap.TASK_DIR/'F1-01.json').write_text(json.dumps(task()))
            result=subprocess.run([sys.executable,str(ROOT/'tools/roadmap.py'),'--root',str(root),
                'claim','F1-01','--owner','agent-a','--machine','linux-a','--branch','work/F1-01/agent-a'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            text=(root/roadmap.ROADMAP).read_text()
            self.assertIn('agent-a / linux-a /',text)
            self.assertIn('| leased |',text)
            result=subprocess.run([sys.executable,str(ROOT/'tools/roadmap.py'),'--root',str(root),'check'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
