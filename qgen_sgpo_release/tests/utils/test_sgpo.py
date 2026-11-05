from __future__ import annotations

import __future__
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from prepare_roll import installed_roll_root, verify_roll
ROLL_ROOT = verify_roll(Path(os.environ["ROLL_ROOT"]) if "ROLL_ROOT" in os.environ else installed_roll_root())
CONFIG_DIR = ROOT / "examples/recr1-qgen-amazon"
SPEC = importlib.util.spec_from_file_location("sgpo_test_module", ROOT / "qgen_sgpo/sgpo.py")
SGPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SGPO)
ENV = {
    "SGPO_ENABLE": "1", "SGPO_C_MAX": "4", "SGPO_C_MIN": "1",
    "SGPO_ANNEAL_STEPS": "200", "SGPO_TOKEN_RATIO": "0", "SGPO_TOKEN_GRAD": "1",
    "SGPO_SIGNAL": "confidence",
    "QGEN_CORPUS": "./data/corpus.jsonl",
}


def load_definitions(path: str, names: set[str], scope: dict) -> None:
    source = (ROLL_ROOT if path.startswith("roll/") else ROOT) / path
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise AssertionError(f"源文件定义缺失：{path}, {names}")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), scope)


def actor_scope() -> dict:
    scope = {"torch": torch, "np": np, "BaseActorWorker": object}
    load_definitions("roll/utils/functionals.py", {"masked_mean", "masked_sum", "agg_loss", "compute_approx_kl"}, scope)
    load_definitions("roll/pipeline/agentic/utils.py", {"compute_segment_masked_mean"}, scope)
    load_definitions("roll/utils/train_infer_corrections.py", {
        "_compute_all_granularity", "_infer_global_valid_samples_from_mask",
        "_infer_global_valid_tokens_from_mask", "compute_train_infer_correction",
    }, scope)
    load_definitions("roll/pipeline/rlvr/actor_worker.py", {"ActorWorker"}, scope)
    return scope


SCOPE = actor_scope()


class TestSGPO(unittest.TestCase):

    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, ENV)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        modules = patch.dict(sys.modules, {"qgen_sgpo.sgpo": SGPO})
        modules.start()
        self.addCleanup(modules.stop)
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        self.values = torch.tensor([[-1., -2., -4., -8.], [-3., -1., -7., -9.], [-2., -8., -8., -8.]], dtype=torch.float64)
        self.advantages = torch.tensor([[1.], [-2.], [0.]], dtype=torch.float64).expand_as(self.values)

    def run_actor(self, step: int = 0, values: torch.Tensor = None, old: torch.Tensor = None,
                  final_mask: torch.Tensor = None, use_kl: bool = True, meta_step: bool = True) -> tuple:
        x = self.values.clone().requires_grad_() if values is None else values
        old = x.detach().clone() if old is None else old
        final_mask = self.mask if final_mask is None else final_mask
        worker = SCOPE["ActorWorker"]()
        worker.strategy = SimpleNamespace(op_compute_log_probs=lambda **kw: kw["logits"])
        worker.get_old_log_probs_with_cache = lambda data, log_probs: old
        worker.pipeline_config = SimpleNamespace(
            enable_old_logprobs_recompute=False,
            train_infer_correction=SimpleNamespace(is_weight=SimpleNamespace(enabled=False), filters=[]),
            importance_sampling="token", pg_clip=0.2, use_pg_clip_range=False,
            dual_clip_loss=False, use_kl_loss=use_kl, kl_loss_coef=0.001,
            rl_loss_coef=1.0, entropy_loss_coef=0.0, loss_agg_mode="seq-mean-token-mean",
            difficulty_loss_weight=False, length_loss_weight=False,
        )
        data = SimpleNamespace(batch={
            "response_mask": torch.cat([torch.zeros(3, 1, dtype=torch.long), self.mask], dim=1),
            "final_response_mask": final_mask, "input_ids": torch.zeros(3, 5, dtype=torch.long),
            "ref_log_probs": self.values.detach() + 0.15, "advantages": self.advantages,
        }, non_tensor_batch={}, meta_info={
            "batch_num_tokens": {"response_mask": self.mask.sum(), "final_response_mask": final_mask.sum()},
            "global_valid_samples": {"response_mask": 3, "final_response_mask": (final_mask.sum(-1) > 0).sum()},
        })
        if meta_step:
            data.meta_info["global_step"] = step
        loss, metrics = worker.loss_func(data, x)
        gradient = torch.autograd.grad(loss, x)[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        return loss.detach(), gradient, metrics

    def test_c1_matches_grpo_full_loss_and_gradient(self) -> None:
        os.environ["SGPO_C_MAX"] = "1"
        sgpo = self.run_actor()
        os.environ["SGPO_ENABLE"] = "0"
        grpo = self.run_actor()
        torch.testing.assert_close(sgpo[0], grpo[0], rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(sgpo[1], grpo[1], rtol=1e-7, atol=1e-9)

    def test_c_changes_actual_loss_gradient(self) -> None:
        gradients = []
        for c in (1, 2, 4):
            os.environ["SGPO_C_MAX"] = str(c)
            loss, gradient, metrics = self.run_actor()
            gradients.append(gradient)
            self.assertAlmostEqual(metrics["actor/ratio_mean@sum"], 1.0, places=7)
            self.assertEqual(metrics["actor/ppo_ratio_clipfrac@sum"], 0.0)
            self.assertEqual(metrics["sgpo/token_grad"], 1.0)
        self.assertGreater((gradients[1] - gradients[0]).norm().item(), 0.1)
        self.assertGreater((gradients[2] - gradients[1]).norm().item(), 0.1)
        print("实际 actor loss 梯度差范数：c2-c1=%.6f，c4-c1=%.6f" % (
            (gradients[1] - gradients[0]).norm(), (gradients[2] - gradients[0]).norm()))

    def test_legacy_cancellation_is_reproduced(self) -> None:
        os.environ["SGPO_TOKEN_GRAD"] = "0"
        os.environ["SGPO_C_MAX"] = "1"
        baseline = self.run_actor()[1]
        os.environ["SGPO_C_MAX"] = "4"
        legacy = self.run_actor()[1]
        torch.testing.assert_close(legacy, baseline, rtol=1e-7, atol=1e-9)

    def test_on_policy_token_ratio_matches_corrected_gradient(self) -> None:
        corrected = self.run_actor()
        os.environ.update(SGPO_TOKEN_GRAD="0", SGPO_TOKEN_RATIO="1")
        token = self.run_actor()
        torch.testing.assert_close(corrected[1], token[1])

    def test_off_policy_forward_and_local_reference_gradient(self) -> None:
        for offset in (-0.5, 0.05, 0.5):
            with self.subTest(offset=offset):
                x = self.values.clone().requires_grad_()
                old = x.detach() - offset
                a, ratio, _ = SGPO.compute_sgpo(x, old, self.advantages, self.mask, 0)
                lengths = self.mask.sum(-1, keepdim=True)
                seq = (((x - old) * self.mask).sum(-1, keepdim=True) / lengths).exp()
                expected = seq.detach() * (x - x.detach()).exp()
                expected = expected * self.mask + (1 - self.mask)
                torch.testing.assert_close(ratio, expected)
                gradients = []
                for r in (ratio, expected):
                    mat = -torch.minimum(r * a, r.clamp(0.8, 1.2) * a)
                    loss = SCOPE["agg_loss"](mat, self.mask, "seq-mean-token-mean", global_valid_samples=3)
                    gradients.append(torch.autograd.grad(loss, x, retain_graph=True)[0])
                torch.testing.assert_close(gradients[0], gradients[1])
                active = ~(((ratio > 1.2) & (a > 0)) | ((ratio < 0.8) & (a < 0)))
                analytic = -ratio.detach() * a * active * self.mask / lengths / (3 + 1e-8)
                torch.testing.assert_close(gradients[0], analytic, rtol=1e-7, atol=1e-9)
                actual = self.run_actor(old=old, use_kl=False)[1]
                torch.testing.assert_close(actual, analytic, rtol=1e-7, atol=1e-9)

    def test_padding_and_dropped_sample_have_zero_gradient(self) -> None:
        final_mask = self.mask.clone()
        final_mask[1] = 0
        _, gradient, _ = self.run_actor(final_mask=final_mask)
        self.assertEqual(gradient[final_mask == 0].abs().max().item(), 0)
        changed = self.values.clone()
        changed[self.mask == 0] = -1000
        changed.requires_grad_()
        other = self.run_actor(values=changed, final_mask=final_mask)[1]
        torch.testing.assert_close(gradient, other)

    def test_advantage_matches_reference_and_preserves_sequence_mean(self) -> None:
        a, _, _ = SGPO.compute_sgpo(self.values, self.values, self.advantages, self.mask, 0)
        for row in range(3):
            valid = self.mask[row].bool()
            logp = self.values[row, valid]
            source = self.advantages[row, valid].mean()
            centered = logp - logp.mean()
            scale = source.abs() * torch.sqrt(3 / (centered.square().mean() + 1e-8))
            expected = source + source.sign() * scale * centered
            torch.testing.assert_close(a[row, valid], expected)
            torch.testing.assert_close(a[row, valid].mean(), source)
        self.assertTrue((a[self.mask == 0] == 0).all())

    def test_advantage_and_old_policy_are_stop_gradient(self) -> None:
        x = self.values.clone().requires_grad_()
        old = x.detach().clone().requires_grad_()
        advantage = self.advantages.clone().requires_grad_()
        a, ratio, _ = SGPO.compute_sgpo(x, old, advantage, self.mask, 0)
        self.assertFalse(a.requires_grad)
        grads = torch.autograd.grad((ratio * a).sum(), (x, old, advantage), allow_unused=True)
        self.assertIsNone(grads[1])
        self.assertIsNone(grads[2])

    def test_empty_single_token_and_constant_sequence(self) -> None:
        for mask in (torch.zeros_like(self.mask), self.mask):
            x = torch.full_like(self.values, -2, requires_grad=True)
            a, ratio, metrics = SGPO.compute_sgpo(x, x.detach(), self.advantages, mask, 0)
            self.assertTrue(torch.isfinite(a).all())
            self.assertEqual(metrics["sgpo/advantage_shift"], 0)
            gradient = torch.autograd.grad((ratio * a).sum(), x)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[mask == 0] == 0).all())

    def test_anneal_uses_training_step_not_call_count(self) -> None:
        for c in (2, 4):
            for n in (200, 600):
                os.environ.update(SGPO_C_MAX=str(c), SGPO_ANNEAL_STEPS=str(n))
                for step in (0, 7, 100, 199, 200, 599, 600):
                    with self.subTest(c=c, n=n, step=step):
                        for _ in range(32):
                            _, _, m = SGPO.compute_sgpo(self.values, self.values, self.advantages, self.mask, step)
                            self.assertAlmostEqual(m["sgpo/c_n"], max(1, c - (c - 1) * step / n))
                            self.assertEqual(m["sgpo/global_step"], step)
                            self.assertEqual(m["sgpo/lambda_k_mean"] > 0, step < n)
                os.environ["SGPO_ENABLE"] = "1"
                retired = self.run_actor(step=n)
                os.environ["SGPO_ENABLE"] = "0"
                baseline = self.run_actor(step=n)
                torch.testing.assert_close(retired[1], baseline[1], rtol=1e-7, atol=1e-9)

    def test_actor_requires_global_step(self) -> None:
        with self.assertRaises(KeyError):
            self.run_actor(meta_step=False)

    def test_float32_and_model_parameter_gradients(self) -> None:
        for dtype in (torch.float32, torch.float64):
            parameter = torch.arange(24, dtype=dtype).reshape(3, 4, 2).div(10).requires_grad_()
            gradients = []
            for c in (1, 4):
                os.environ["SGPO_C_MAX"] = str(c)
                logp = torch.log_softmax(parameter, -1)[..., 0] * self.values.to(dtype)
                gradient = self.run_actor(values=logp)[1]
                gradients.append(torch.autograd.grad(logp, parameter, gradient, retain_graph=True)[0])
            self.assertGreater((gradients[1] - gradients[0]).norm().item(), 0.01)

    def test_invalid_config_fails_closed(self) -> None:
        for override in ({"SGPO_C_MIN": "0.5"}, {"SGPO_C_MAX": "nan"},
                         {"SGPO_ANNEAL_STEPS": "0"}, {"SGPO_TOKEN_RATIO": "1"},
                         {"SGPO_TOKEN_GRAD": "typo"}, {"SGPO_SIGNAL": "typo"}):
            with self.subTest(override=override), patch.dict(os.environ, override):
                with self.assertRaises(ValueError):
                    SGPO.compute_sgpo(self.values, self.values, self.advantages, self.mask, 0)
        for step in (-1, 1.5):
            with self.assertRaises(ValueError):
                SGPO.compute_sgpo(self.values, self.values, self.advantages, self.mask, step)

    def entry_scope(self) -> dict:
        scope = {"os": os}
        path = ROOT / "examples/start_rl_qgen.py"
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_WORKER_ENV_PREFIXES" for t in node.targets):
                scope["_WORKER_ENV_PREFIXES"] = ast.literal_eval(node.value)
        load_definitions("examples/start_rl_qgen.py", {"set_job_env", "propagate_env_to_workers"}, scope)
        return scope

    def test_entry_rejects_invalid_method_and_reward_protocol(self) -> None:
        scope = self.entry_scope()
        for override in ({'SGPO_ENABLE': 'typo'}, {'QGEN_TOP_K': '50'},
                         {'QGEN_MATCH_MODE': 'lucene_bool'}):
            with self.subTest(override=override), patch.dict(os.environ, override):
                with self.assertRaises(ValueError):
                    scope['set_job_env']('')

    def test_env_reaches_every_worker_and_rejects_legacy(self) -> None:
        scope = self.entry_scope()
        self.assertEqual(scope['_WORKER_ENV_PREFIXES'], ('QGEN_', 'SGPO_'))
        scope["set_job_env"](",".join(f"{k}={v}" for k, v in ENV.items()))
        targets = [SimpleNamespace(system_envs={"EXISTING": "keep"}) for _ in range(5)]
        cfg = SimpleNamespace(actor_train=targets[0], actor_infer=targets[1], reference=targets[2],
                              critic=targets[3], rewards={"qgen": targets[4]})
        scope["propagate_env_to_workers"](cfg)
        for worker in targets:
            for key, value in ENV.items():
                self.assertEqual(worker.system_envs[key], value)
            self.assertEqual(worker.system_envs["EXISTING"], "keep")
        with self.assertRaises(ValueError):
            scope["set_job_env"]("SGPO_TOKEN_GRAD=0")


class TestReleaseContract(unittest.TestCase):
    def test_three_arm_shared_config(self):
        configs = []
        names = ('qgen_grpo_3b_2gpu', 'qgen_gspo_3b_2gpu_s42', 'qgen_sgpo_tg_c4_a200_3b_2gpu')
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            for name in names:
                cfg = OmegaConf.to_container(compose(config_name=name), resolve=True)
                self.assertEqual((cfg['seed'], cfg['max_steps']), (42, 600))
                self.assertEqual((cfg['rollout_batch_size'], cfg['num_return_sequences_in_group']), (32, 8))
                self.assertEqual(cfg['actor_train']['training_args']['gradient_accumulation_steps'], 32)
                self.assertEqual(cfg['checkpoint_config']['type'], 'file_system')
                self.assertEqual(cfg['rewards']['qgen']['worker_cls'], 'qgen_sgpo.reward_worker.QgenBM25RewardWorker')
                for role in ('actor_train', 'actor_infer', 'reference'):
                    self.assertEqual(cfg[role]['device_mapping'], 'list(range(0,2))')
                for field in ('exp_name', 'tracker_kwargs', 'importance_sampling'):
                    cfg.pop(field, None)
                cfg['checkpoint_config']['output_dir'] = '<arm-specific-output>'
                configs.append(cfg)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])

    def eval_module(self):
        spec = importlib.util.spec_from_file_location('release_eval', ROOT / 'examples/start_eval_qgen.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_metrics_use_all_samples_at_nine_depths(self):
        module = self.eval_module()
        records = [{'rank': r, 'format_correct': r > 0, 'total': 0, 'format': 0,
                    'answer': 0, 'n_retrieved': 0, 'query': None} for r in (1, 2, 0, 101)]
        metrics = module.compute_metrics(records, 100)
        import math
        self.assertAlmostEqual(metrics['ndcg@1'], 0.25)
        self.assertAlmostEqual(metrics['mrr@2'], 0.375)
        self.assertAlmostEqual(metrics['ndcg@2'], (1 + 1 / math.log2(3)) / 4)
        self.assertEqual(metrics['n'], 4)
        self.assertEqual(metrics['hit@1'], 0.25)
        self.assertEqual(metrics['hit@2'], 0.5)
        self.assertEqual(metrics['hit@100'], 0.5)
        self.assertEqual(metrics['hit_rate'], metrics['hit@100'])
        self.assertEqual(metrics['mrr'], metrics['mrr@100'])
        for metric in ('ndcg', 'mrr', 'hit'):
            previous = 0.0
            for k in (1, 2, 3, 5, 10, 20, 30, 50, 100):
                key = f'{metric}@{k}'
                self.assertIn(key, metrics)
                self.assertIn(key, module.METRIC_ORDER)
                self.assertGreaterEqual(metrics[key], previous)
                previous = metrics[key]
        shallow = module.compute_metrics(records, 20)
        self.assertNotIn('hit@30', shallow)
        self.assertEqual(shallow['hit_rate'], shallow['hit@20'])

    def test_metrics_no_hits_or_empty_records(self):
        module = self.eval_module()
        records = [{'rank': 0, 'format_correct': False, 'total': -2, 'format': -2,
                    'answer': 0, 'n_retrieved': 0, 'query': None} for _ in range(2)]
        metrics = module.compute_metrics(records, 100)
        self.assertEqual(metrics['n'], 2)
        self.assertEqual(metrics['hit_rate'], 0)
        for metric in ('ndcg', 'mrr', 'hit'):
            for k in (1, 2, 3, 5, 10, 20, 30, 50, 100):
                self.assertEqual(metrics[f'{metric}@{k}'], 0)
        self.assertEqual(module.compute_metrics([], 100), {'n': 0})

    def test_public_pipeline_preserves_preformatted_prompts(self):
        seen = []

        def tokenizer(texts):
            seen.extend(texts)
            return {'input_ids': [[i] for i in range(len(texts))]}

        scope = {'get_chat_template': lambda name, tok:
                 lambda messages: self.fail('预模板化 prompt 不应再次套 chat template')}
        load_definitions('roll/pipeline/rlvr/rlvr_pipeline.py', {'get_encode_function'}, scope)
        prompts = ['<|im_start|>user\n第一条<|im_end|>\n<|im_start|>assistant\n',
                   '<|im_start|>user\n第二条<|im_end|>\n<|im_start|>assistant\n']
        encode = scope['get_encode_function']('qwen2_5', tokenizer, SimpleNamespace(prompt='prompt'))
        result = encode({'prompt': prompts})
        self.assertEqual(seen, prompts)
        self.assertEqual(result['input_ids'], [[0], [1]])

    def test_eval_missing_local_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.eval_module().resolve_model_in_subprocess(ROOT / 'missing-checkpoint', ROOT)

    def test_scorer_is_independent_of_roll(self):
        module, retriever = self.eval_module().load_scorer(str(ROOT / 'data/corpus.jsonl'), 100, 'or_all')
        self.assertEqual(len(retriever.doc_ids), 3686)
        self.assertEqual(module.__name__, 'qgen_sgpo.qgen_bm25')

    def test_offline_wrapper_and_reward_semantics(self):
        tools = ROOT / 'examples/recr1-qgen-amazon/tools'
        with patch.object(sys, 'path', [str(tools), *sys.path]):
            import importlib
            import bm25
            module = importlib.import_module('selfcheck_qgen')
            self.assertEqual(Path(bm25._REAL), ROOT / 'qgen_sgpo/qgen_bm25.py')
            self.assertEqual(Path(module.DEFAULT_DATA), ROOT / 'data')
            index = bm25.BM25Index.from_corpus(str(ROOT / 'data/corpus.jsonl'))
            module._fails.clear()
            module.check_reward_semantics(index)
            self.assertEqual(module._fails, [])

    def test_reward_worker_preserves_response_rewards(self):
        from qgen_sgpo.qgen_bm25 import BM25Index, compute_score
        scope = {
            'torch': torch, 'Worker': object,
            'register': lambda **kwargs: lambda fn: fn,
            'Dispatch': SimpleNamespace(ONE_TO_ALL=0, DP_MP_COMPUTE=1),
            'logger': SimpleNamespace(info=lambda message: None),
            'compute_score': compute_score,
            'DataProto': SimpleNamespace(from_dict=lambda tensors: tensors),
        }
        load_definitions('qgen_sgpo/reward_worker.py', {'QgenBM25RewardWorker'}, scope)
        cls = scope['QgenBM25RewardWorker']
        worker = cls.__new__(cls)
        texts = ['<think>r</think><answer>{"query": "steel wrench"}</answer>',
                 '<think>r</think><answer>{"query": "zzzqqqxxx"}</answer>',
                 '<answer>{"query": "steel wrench"}</answer>']
        worker.tokenizer = SimpleNamespace(batch_decode=lambda ids, **kwargs: texts)
        worker.retriever = BM25Index().build(['gold'], ['steel wrench'])
        worker.top_k, worker.mode = 100, 'or_all'
        responses = torch.zeros(3, 5, dtype=torch.long)
        data = SimpleNamespace(batch={'responses': responses},
                               non_tensor_batch={'ground_truth': ['gold'] * 3})
        output = worker.compute_rewards(data)
        self.assertEqual(set(output), {'token_level_rewards', 'response_level_rewards', 'scores'})
        torch.testing.assert_close(output['response_level_rewards'],
                                   torch.tensor([1.1, 0, -2], dtype=torch.float16))
        torch.testing.assert_close(output['scores'], torch.tensor([1, 0, 0], dtype=torch.float16))
        torch.testing.assert_close(output['token_level_rewards'],
                                   torch.zeros_like(responses, dtype=torch.float16))

    def test_data_export_preserves_shipped_prompt(self):
        import csv
        import json
        import tempfile
        path = ROOT / 'examples/recr1-qgen-amazon/tools/export_rl_jsonl.py'
        spec = importlib.util.spec_from_file_location('release_data_export', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (ROOT / 'data/rl_train.jsonl').open() as stream:
            shipped = json.loads(next(stream))
        titles = [line.split('. ', 1)[1] for line in shipped['prompt'].splitlines()
                  if line[:1].isdigit() and '. ' in line]
        self.assertEqual(module.build_prompt(titles, 'industrial and scientific'), shipped['prompt'])
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'source.csv', Path(temp) / 'train.jsonl'
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['history_item_title', 'item_id'])
                writer.writeheader()
                writer.writerow({'history_item_title': repr(titles), 'item_id': shipped['ground_truth']})
            module.export(source, target, 'industrial and scientific')
            self.assertEqual(json.loads(target.read_text()), shipped)

    def test_data_manifest_order_is_irrelevant(self):
        import verify_release
        counts = {'data/rl_test.jsonl': 4533, 'data/rl_val.jsonl': 4532,
                  'data/rl_train.jsonl': 36259, 'data/corpus.jsonl': 3686}
        verify_release.verify_data(counts)
        with self.assertRaises(RuntimeError):
            verify_release.verify_data({'data/corpus.jsonl': 3686})

    def test_framework_rejects_wrong_commit_partial_patch_and_injection(self):
        import json
        import prepare_roll
        with self.assertRaises(RuntimeError):
            prepare_roll.verify_roll(ROOT / 'vendor/roll')
        with patch.object(prepare_roll, 'git', return_value='incompatible-commit'):
            with self.assertRaises(RuntimeError):
                prepare_roll.verify_roll(ROLL_ROOT)
        manifest = json.loads(prepare_roll.MANIFEST.read_text())
        actor = 'roll/pipeline/rlvr/actor_worker.py'
        real_digest = prepare_roll.digest
        with patch.object(prepare_roll, 'digest', side_effect=lambda p:
                          manifest['files'][actor]['before'] if Path(p) == ROLL_ROOT / actor
                          else real_digest(p)):
            with self.assertRaises(RuntimeError):
                prepare_roll.verify_roll(ROLL_ROOT)
        real_git = prepare_roll.git
        with patch.object(prepare_roll, 'git', side_effect=lambda root, *args:
                          'roll/injected.py' if args[0] == 'ls-files' else real_git(root, *args)):
            with self.assertRaises(RuntimeError):
                prepare_roll.verify_roll(ROLL_ROOT)

    def test_eval_mock_io_and_incomplete_generation(self):
        import contextlib
        import io
        import json
        import tempfile
        module = self.eval_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'checkpoint'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            weights = model / 'model.safetensors'
            weights.write_bytes(b'fixture')
            data = root / 'data.jsonl'
            data.write_text(json.dumps({'prompt': 'fixture', 'ground_truth': '1'}) + '\n')
            vllm = ModuleType('vllm')
            response = '<think>r</think><answer>{"query": "Stanley TRA708T"}</answer>'
            output = SimpleNamespace(outputs=[SimpleNamespace(text=response, token_ids=[1, 2])])
            vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
            generator = SimpleNamespace(generate=lambda prompts, params: [output])
            vllm.LLM = lambda **kwargs: generator
            argv = ['start_eval_qgen.py', '--model_name', str(model), '--data_file', str(data),
                    '--corpus', str(ROOT / 'data/corpus.jsonl'), '--out_dir', str(root / 'out')]
            with patch.dict(sys.modules, {'vllm': vllm}), patch.dict(os.environ, {'RANK': '0'}), \
                    patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as output_log:
                module.main()
                metrics = json.loads((root / 'out/metrics_checkpoint-599.json').read_text())
                self.assertEqual(metrics['n'], 1)
                self.assertIn('ndcg@100', metrics)
                self.assertIn('hit@100', metrics)
                self.assertEqual(metrics['hit_rate'], metrics['hit@100'])
                for k in (1, 2, 3, 5, 10, 20, 30, 50, 100):
                    self.assertIn(f'hit@{k}', output_log.getvalue())
                self.assertEqual(weights.read_bytes(), b'fixture')
                with self.assertRaises(FileExistsError):
                    module.main()
                argv[-1] = str(root / 'incomplete')
                generator.generate = lambda prompts, params: []
                with self.assertRaises(RuntimeError):
                    module.main()
                self.assertFalse((root / 'incomplete/metrics_all.json').exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
