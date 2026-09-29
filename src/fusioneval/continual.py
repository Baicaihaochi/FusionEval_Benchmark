from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import yaml
from .config import _SLUG, _mapping, _overlap, _read_yaml, _resolve, normalize_runtime, load_config
from .continual_rng import expected_generator_device, read_stage_rng, sha256_of_state
from .engine import run as run_method
from .errors import ConfigError, ExecutionError
from .methods.fields import choice, optional_bool, optional_mapping, parameters as exact_parameters, positive_int, scale, unit_interval
from .model_profile import load_model_profile
_RUN_KEYS = {'schema_version', 'common', 'method', 'parameters', 'order', 'output', 'data'}
_COMMON_KEYS = {'schema_version', 'model_profile', 'base', 'experts', 'order', 'initial', 'output_root', 'runtime'}
_METHODS = {'recursive_average', 'regmean', 'task_arithmetic', 'ties', 'della'}
_TWO_INPUT_METHODS = {'ties', 'della'}
_PLAN_ONLY_METHODS = frozenset()
_RNG_METHODS = {'della'}
_EXTENSION_PROTOCOL = 'fusioneval-continual-fixed-anchor-sparse-v1'
_SEQUENCE_PROTOCOL = 'fusioneval-continual-sequence-direct-v2'
_STAGE_PROTOCOL = 'fusioneval-continual-stage-v2'
_RECURSIVE_AVERAGE_DIVISOR = 2
_RECURSIVE_AVERAGE_EXPECTED = (1.0 / 16, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2)
_EXPECTED_INITIAL_EXPERTS = 5
_STATISTICS_METHODS = {'regmean'}

@dataclass(frozen=True)
class ContinualExpert:
    id: str
    path: Path
    data: Optional[Path] = None

@dataclass(frozen=True)
class ContinualConfig:
    path: Path
    common_path: Path
    method: str
    parameters: Dict[str, Any]
    output_name: str
    output_root: Path
    base: Path
    initial_id: str
    experts: Tuple[ContinualExpert, ...]
    order: Tuple[str, ...]
    runtime: Dict[str, Any]
    model_profile_path: Path

    @property
    def run_root(self) -> Path:
        return self.output_root / self.output_name

    @property
    def initial(self) -> ContinualExpert:
        return next((item for item in self.experts if item.id == self.initial_id))

def _path_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError('{} must be a non-empty path string'.format(location))
    return value

def load_continual_config(path: Path) -> ContinualConfig:
    path = path.resolve()
    raw = _read_yaml(path, 'continual run config')
    extra = set(raw) - _RUN_KEYS
    if extra:
        raise ConfigError('unsupported continual run key(s): {}'.format(', '.join(sorted(extra))))
    if raw.get('schema_version') != 1:
        raise ConfigError('continual run schema_version must be 1')
    common_value = _path_string(raw.get('common'), 'continual common')
    common_path = _resolve(common_value, path.parent)
    common = _read_yaml(common_path, 'continual common config')
    common_extra = set(common) - _COMMON_KEYS
    if common_extra:
        raise ConfigError('unsupported continual common key(s): {}'.format(', '.join(sorted(common_extra))))
    if common.get('schema_version') != 1:
        raise ConfigError('continual common schema_version must be 1')
    method = raw.get('method')
    if method not in _METHODS:
        raise ConfigError('continual method must be one of {}'.format(', '.join(sorted(_METHODS))))
    if method == 'recursive_average':
        configured = dict(exact_parameters(raw.get('parameters', {}), ()))
    elif method == 'regmean':
        value = optional_mapping(raw.get('parameters', {}), ('alpha', 'examples', 'leftover_edge', 'leftover_1d'))
        if 'alpha' not in value:
            raise ConfigError('missing method parameter(s): alpha')
        configured = {'alpha': unit_interval(value['alpha'], 'alpha'), 'examples': positive_int(value['examples'], 'examples') if 'examples' in value else 256, 'leftover_edge': choice(value, 'leftover_edge', 'mean', ('base', 'mean')), 'leftover_1d': choice(value, 'leftover_1d', 'mean', ('base', 'mean'))}
    elif method in _TWO_INPUT_METHODS:
        required = {'ties': ('density', 'scale', 'merge_func', 'exclude_edge'), 'dare': ('drop_rate', 'scale', 'seed', 'exclude_edge'), 'della': ('drop_rate', 'window', 'scale', 'seed', 'exclude_edge')}[method]
        value = exact_parameters(raw.get('parameters', {}), required)
        from .methods.ties import normalize as normalize_ties
        from .methods.dare import normalize as normalize_dare
        from .methods.della import normalize as normalize_della
        exclude_edge = optional_bool(value, 'exclude_edge', False)
        if exclude_edge:
            raise ConfigError('continual sparse methods require exclude_edge=false; an edge fallback needs a separate defined protocol')
        if method == 'ties':
            if value['merge_func'] != 'mean':
                raise ConfigError('continual ties requires merge_func=mean')
            normalized = normalize_ties(value)
        else:
            seed = value['seed']
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ConfigError('method.parameters.seed must be a non-negative integer')
            static = {k: v for k, v in value.items() if k != 'seed'}
            if method == 'della':
                static.pop('exclude_edge')
                normalized = normalize_della(static)
            else:
                normalized = normalize_dare(static)
            normalized.update(seed=seed, exclude_edge=exclude_edge)
        configured = {key: normalized[key] for key in required}
    else:
        value = exact_parameters(raw.get('parameters', {}), ('scale',))
        configured = {'scale': scale(value['scale'])}
    output_name = raw.get('output')
    if not isinstance(output_name, str) or not _SLUG.fullmatch(output_name):
        raise ConfigError('continual output must be a short path-safe slot name')
    base = _resolve(_path_string(common.get('base'), 'continual common.base'), common_path.parent)
    experts_raw = common.get('experts')
    if not isinstance(experts_raw, list) or not experts_raw:
        raise ConfigError('continual common.experts must be a non-empty list')
    experts: List[ContinualExpert] = []
    seen_ids = set()
    for index, item in enumerate(experts_raw):
        item = _mapping(item, 'continual common.experts[{}]'.format(index))
        item_extra = set(item) - {'id', 'path'}
        if item_extra:
            raise ConfigError('unsupported continual expert key(s): {}'.format(', '.join(sorted(item_extra))))
        expert_id = item.get('id')
        if not isinstance(expert_id, str) or not _SLUG.fullmatch(expert_id):
            raise ConfigError('each continual expert id must be a short path-safe slug')
        if expert_id in seen_ids:
            raise ConfigError('duplicate continual expert id: {}'.format(expert_id))
        expert_path = _resolve(_path_string(item.get('path'), 'continual expert path'), common_path.parent)
        seen_ids.add(expert_id)
        experts.append(ContinualExpert(expert_id, expert_path))
    initial_id = common.get('initial')
    if not isinstance(initial_id, str) or not _SLUG.fullmatch(initial_id):
        raise ConfigError('continual common.initial must name the expert that provides M0; every method starts from the complete initial expert and fuses only the later arrivals')
    if initial_id not in seen_ids:
        raise ConfigError('continual common.initial must name one of the declared experts')
    if method in _STATISTICS_METHODS:
        data_raw = _mapping(raw.get('data'), 'continual {} data'.format(method))
        if set(data_raw) != seen_ids:
            raise ConfigError('continual {} data must contain every expert id exactly once'.format(method))
        experts = [ContinualExpert(item.id, item.path, _resolve(_path_string(data_raw[item.id], 'continual {} data.{}'.format(method, item.id)), path.parent)) for item in experts]
    elif 'data' in raw:
        raise ConfigError('continual data is only supported by Fisher and RegMean')
    order_raw = raw.get('order', common.get('order'))
    if not isinstance(order_raw, list) or not all((isinstance(item, str) for item in order_raw)):
        raise ConfigError('continual order must be a list of expert ids')
    order = tuple(order_raw)
    arrivals = seen_ids - {initial_id}
    if len(order) != len(arrivals) or len(set(order)) != len(order) or set(order) != arrivals:
        raise ConfigError('continual order must list every non-initial expert exactly once; the initial expert is Stage 0 and must not appear in the arrival order')
    source_paths = [base] + [item.path for item in experts]
    if len(set(source_paths)) != len(source_paths):
        raise ConfigError('continual base and expert checkpoint paths must be distinct')
    output_root = _resolve(_path_string(common.get('output_root'), 'continual common.output_root'), common_path.parent)
    run_root = output_root / output_name
    if any((_overlap(run_root, source) for source in source_paths)):
        raise ConfigError('continual output must not overlap a source checkpoint')
    profile_path = _resolve(_path_string(common.get('model_profile'), 'continual common.model_profile'), common_path.parent)
    load_model_profile(profile_path)
    runtime = normalize_runtime(common.get('runtime', {}), 'continual common.runtime')
    if method in {'dare', 'della'}:
        declared_runtime = common.get('runtime', {})
        if 'seed' in declared_runtime and runtime['seed'] != configured['seed']:
            raise ConfigError('continual runtime.seed conflicts with method.parameters.seed')
        runtime['seed'] = configured['seed']
    return ContinualConfig(path=path, common_path=common_path, method=method, parameters=configured, output_name=output_name, output_root=output_root, base=base, initial_id=initial_id, experts=tuple(experts), order=order, runtime=runtime, model_profile_path=profile_path)

def _stage_operation(config: ContinualConfig, stage: int) -> Tuple[str, Dict[str, Any]]:
    if stage < 1:
        raise ConfigError('updates start at 1 (Stage 0 is the initial expert)')
    experts_included = stage + 1
    if config.method == 'recursive_average':
        return ('online_average', {'seen_count': _RECURSIVE_AVERAGE_DIVISOR})
    if config.method == 'regmean':
        return ('continual_regmean', {'alpha': config.parameters['alpha'], 'examples': config.parameters['examples'], 'leftover_edge': config.parameters['leftover_edge'], 'leftover_1d': config.parameters['leftover_1d'], 'experts_included': experts_included})
    if config.method == 'ties':
        return ('continual_ties', {'density': config.parameters['density'], 'scale': config.parameters['scale'], 'merge_func': config.parameters['merge_func'], 'exclude_edge': config.parameters['exclude_edge'], 'experts_included': experts_included, 'participant_count': 2})
    if config.method == 'della':
        return ('continual_della', {'drop_rate': config.parameters['drop_rate'], 'window': config.parameters['window'], 'scale': config.parameters['scale'], 'seed': config.parameters['seed'], 'exclude_edge': config.parameters['exclude_edge'], 'experts_included': experts_included, 'participant_count': 2})
    return ('continual_task_arithmetic', {'scale': config.parameters['scale'], 'experts_included': experts_included})

def _initial_operation(config: ContinualConfig) -> Tuple[str, Dict[str, Any]]:
    if config.method == 'regmean':
        return ('continual_regmean', {'alpha': config.parameters['alpha'], 'examples': config.parameters['examples'], 'leftover_edge': config.parameters['leftover_edge'], 'leftover_1d': config.parameters['leftover_1d'], 'experts_included': 1})
    raise ConfigError('method {} has no stage-0 computation: Stage 0 is the initial expert itself'.format(config.method))

def _effective_expert_weights(config: ContinualConfig) -> Optional[Dict[str, Any]]:
    ids = [config.initial_id] + list(config.order)
    if config.method == 'recursive_average':
        weights = {config.initial_id: 1.0}
        for expert_id in config.order:
            weights = {key: 0.5 * value for key, value in weights.items()}
            weights[expert_id] = 0.5
        values = [weights[expert_id] for expert_id in ids]
    elif config.method == 'task_arithmetic':
        values = [1.0] + [float(config.parameters['scale'])] * len(config.order)
    else:
        return None
    result = {'order': ids, 'values': values, 'sum': sum(values)}
    if config.method == 'recursive_average' and len(ids) == _EXPECTED_INITIAL_EXPERTS:
        expected = list(_RECURSIVE_AVERAGE_EXPECTED)
        if any((abs(left - right) > 1e-12 for left, right in zip(values, expected))):
            raise ExecutionError('recursive_average coefficients {} != section-0 {}'.format(values, expected))
        result['section0_expected'] = expected
        result['coefficients_confirmed'] = True
    if config.method == 'task_arithmetic' and len(ids) == _EXPECTED_INITIAL_EXPERTS:
        expected = [1.0] + [float(config.parameters['scale'])] * len(config.order)
        if any((abs(left - right) > 1e-12 for left, right in zip(values, expected))):
            raise ExecutionError('task_arithmetic coefficients {} != expected {}'.format(values, expected))
        result['section0_expected'] = expected
        result['coefficients_confirmed'] = True
        result['note'] = 'the initial expert carries coefficient 1; every later arrival carries the task-arithmetic scale, because each increment is taken relative to the fixed common base and never relative to the initial expert'
    return result

def compile_continual_plan(config: ContinualConfig) -> Dict[str, Any]:
    by_id = {item.id: item for item in config.experts}
    initial = config.initial
    statistics_method = config.method in _STATISTICS_METHODS
    if statistics_method:
        operation, initial_parameters = _initial_operation(config)
        initial_source_roles = {'incoming': str(initial.path), 'incoming_data': str(initial.data)}
        initial_formula = 'theta_0=theta_initial (weights unmodified); statistics initialised on the initial expert; no fusion, no scaling, no solve'
    else:
        operation = 'initial_expert'
        initial_parameters = None
        initial_source_roles = {'incoming': str(initial.path)}
        initial_formula = 'theta_0=theta_initial; M0 is the initial expert itself'
    initial_block = {'observation': 0, 'domain': initial.id, 'kind': 'statistics_initialisation' if statistics_method else 'initial_expert', 'operation': operation, 'executes': statistics_method, 'model': str(initial.path), 'output': str(config.run_root / 'stages' / 's00'), 'formula': initial_formula, 'weights': 'verbatim copy of the initial expert weights' if statistics_method else 'not materialised: the initial expert checkpoint is used in place', 'source_roles': initial_source_roles}
    if initial_parameters is not None:
        initial_block['parameters'] = initial_parameters
        initial_block['experts_included'] = 1
    stages = []
    for stage, expert_id in enumerate(config.order, 1):
        incoming = by_id[expert_id]
        operation, method_parameters = _stage_operation(config, stage)
        output = config.run_root / 'stages' / 's{:02d}'.format(stage)
        if stage == 1:
            previous = config.run_root / 'stages' / 's00' if statistics_method else initial.path
        else:
            previous = config.run_root / 'stages' / 's{:02d}'.format(stage - 1)
        source_roles = {'previous': str(previous), 'incoming': str(incoming.path)}
        if statistics_method:
            if incoming.data is None:
                raise ConfigError('continual {} expert has no calibration data'.format(config.method))
            source_roles['incoming_data'] = str(incoming.data)
        if config.method in {'task_arithmetic'} | _TWO_INPUT_METHODS:
            source_roles['immutable_anchor'] = str(config.base)
        stages.append({'stage': stage, 'domain': expert_id, 'operation': operation, 'experts_included': stage + 1, 'parameters': method_parameters, 'source_roles': source_roles, 'output': str(output)})
    if config.method == 'regmean':
        formula = 'theta_t=solve(shrunk(G_prev)+shrunk(G_in),shrunk(G_prev)@theta_prev+shrunk(G_in)@theta_in);shrunk(G)=alpha*G+(1-alpha)*diag(G); experts_included=t+1'
    elif config.method == 'recursive_average':
        formula = 'theta_0=theta_initial; theta_t=0.5*theta_prev+0.5*theta_incoming for t=1..4'
    elif config.method == 'ties':
        formula = 'theta_0=theta_initial; h=theta_prev-theta_anchor; d=theta_incoming-theta_anchor; theta_t=theta_anchor+scale*ties_density_disjoint_mean(h,d) for t=1..4; the accumulated vector is re-trimmed each stage and votes once'
    elif config.method == 'della':
        formula = 'theta_0=theta_initial; h=theta_prev-theta_anchor; d=theta_incoming-theta_anchor; theta_t=theta_anchor+scale*della_drop_window(h,d) for t=1..4; both inputs are re-sparsified each stage'
    else:
        formula = 'theta_0=theta_initial; theta_t=theta_prev+scale*(theta_incoming-theta_anchor) for t=1..4'
    plan = {'schema_version': 1, 'protocol': _EXTENSION_PROTOCOL if config.method in _TWO_INPUT_METHODS else _SEQUENCE_PROTOCOL, 'numbering': 'section0-observation-0..4', 'method': config.method, 'parameters': config.parameters, 'formula': formula, 'initial': initial_block, 'common_anchor': str(config.base), 'order': list(config.order), 'expert_paths': {item.id: str(item.path) for item in config.experts}, 'model_profile': str(config.model_profile_path), 'runtime': config.runtime, 'observation_points': len(config.order) + 1, 'updates': len(config.order), 'effective_expert_weights': _effective_expert_weights(config), 'access_contract': {'historical_expert_checkpoints': False, 'previous_merged_checkpoint': True, 'incoming_expert_checkpoint': True, 'initial_expert_checkpoint': True, 'stage_0_is_not_an_update': True, 'immutable_common_anchor': config.method in {'task_arithmetic'} | _TWO_INPUT_METHODS}, 'stages': stages, 'output': str(config.run_root), 'config': str(config.path), 'common_config': str(config.common_path)}
    if config.method in _TWO_INPUT_METHODS:
        plan_only = config.method in _PLAN_ONLY_METHODS
        plan['execution'] = {'ready': not plan_only, 'status': 'PLAN_ONLY' if plan_only else 'READY', 'runner': 'fusioneval.methods.continual_{}.kernel'.format(config.method), 'input_mode': 'experts', 'reason': 'continual runner and atomic RNG/resume support are not implemented' if plan_only else 'runner, atomic RNG persistence and numerical/resume verification are implemented; Stage 0 stays the initial expert and every fusion is anchored on the immutable common base'}
        if config.method in _RNG_METHODS:
            plan['rng_contract'] = {'initial_seed': config.runtime['seed'], 'state_device': expected_generator_device('continual_{}'.format(config.method), config.runtime['device']), 'state_location': 'stage manifest diagnostics.rng (atomic stage directory)', 'state_hash': 'sha256 of the raw torch.Generator state blob', 'continuation': 'previous-stage atomic RNG state; never reset per stage', 'seeded_once_at': 'Stage 1', 'implemented': not plan_only}
    if statistics_method:
        plan['expert_data'] = {item.id: str(item.data) for item in config.experts}
        plan['access_contract'].update({'calibration_data': True, 'initial_statistics': True, 'previous_cumulative_state': True})
    return plan

def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.{}.tmp'.format(path.name))
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')

def _write_stage_yaml(path: Path, value: Mapping[str, Any], *, completed: bool) -> None:
    text = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True)
    if completed and path.is_file():
        if path.read_text(encoding='utf-8') != text:
            raise ExecutionError('existing generated stage config differs: {}'.format(path))
        return
    _atomic_text(path, text)

def _generated_stage_configs(config: ContinualConfig, stage_plan: Mapping[str, Any], runtime_extra: Optional[Mapping[str, Any]]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stage = int(stage_plan['stage'])
    roles = stage_plan['source_roles']
    if config.method in _TWO_INPUT_METHODS:
        experts = [{'id': 'previous', 'path': roles['previous']}, {'id': 'incoming', 'path': roles['incoming']}]
    elif config.method == 'task_arithmetic':
        experts = [{'id': 'anchor', 'path': roles['immutable_anchor']}, {'id': 'incoming', 'path': roles['incoming']}]
    else:
        incoming = {'id': 'incoming', 'path': roles['incoming']}
        if config.method in _STATISTICS_METHODS:
            incoming['data'] = roles['incoming_data']
        experts = [incoming]
    common = {'schema_version': 1, 'model_profile': str(config.model_profile_path), 'base': roles['immutable_anchor'] if config.method in _TWO_INPUT_METHODS else roles['previous'], 'experts': experts, 'output_root': str(config.run_root / 'stages'), 'runtime': {**config.runtime, **dict(runtime_extra or {})}}
    run = {'schema_version': 1, 'common': 's{:02d}.common.yaml'.format(stage), 'method': stage_plan['operation'], 'parameters': stage_plan['parameters'], 'output': 's{:02d}'.format(stage)}
    return (common, run)

def _generated_initial_configs(config: ContinualConfig, initial_block: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    roles = initial_block['source_roles']
    common = {'schema_version': 1, 'model_profile': str(config.model_profile_path), 'base': str(config.base), 'initial_model': roles['incoming'], 'experts': [{'id': 'incoming', 'path': roles['incoming'], 'data': roles['incoming_data']}], 'output_root': str(config.run_root / 'stages'), 'runtime': config.runtime}
    run = {'schema_version': 1, 'common': 's00.common.yaml', 'method': initial_block['operation'], 'parameters': initial_block['parameters'], 'output': 's00'}
    return (common, run)

def _initial_stage_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    block = plan['initial']
    return {'stage': 0, 'domain': block['domain'], 'operation': block['operation'], 'experts_included': 1, 'parameters': block['parameters'], 'source_roles': block['source_roles'], 'output': block['output'], 'model': block['model'], 'executes': True}

def _pointer_stage0_receipt(plan: Mapping[str, Any]) -> Dict[str, Any]:
    block = plan['initial']
    return {'schema_version': 1, 'status': 'PASS', 'stage': 0, 'domain': block['domain'], 'operation': 'initial_expert', 'executes': False, 'experts_included': 1, 'model': block['model'], 'output': block['output'], 'weights': block['weights'], 'formula': block['formula'], 'algorithm_runtime_seconds': 0.0}

def _write_pointer_stage0(plan: Mapping[str, Any]) -> Dict[str, Any]:
    receipt = _pointer_stage0_receipt(plan)
    output = Path(receipt['output'])
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'stage0.json'
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    if path.is_file():
        if path.read_text(encoding='utf-8') != text:
            raise ExecutionError('existing stage-0 pointer receipt differs: {}'.format(path))
    else:
        _atomic_text(path, text)
    return receipt

def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError('cannot read {} {}: {}'.format(label, path, exc)) from exc
    if not isinstance(value, dict):
        raise ExecutionError('{} must contain a JSON object: {}'.format(label, path))
    return value

def _receipt_from_manifest(output: Path, stage_plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    timing = manifest.get('timing')
    if not isinstance(timing, Mapping) or 'algorithm_runtime_seconds' not in timing:
        raise ExecutionError('stage manifest has no complete timing record: {}'.format(output))
    receipt = {'schema_version': 1, 'status': 'PASS', 'stage': stage_plan['stage'], 'domain': stage_plan['domain'], 'operation': stage_plan['operation'], 'experts_included': stage_plan['experts_included'], 'parameters': stage_plan['parameters'], 'source_roles': stage_plan['source_roles'], 'output': str(output), 'algorithm_runtime_seconds': timing['algorithm_runtime_seconds']}
    for key in ('model', 'executes'):
        if key in stage_plan:
            receipt[key] = stage_plan[key]
    return receipt

def _load_stage_receipt(output: Path, stage_plan: Mapping[str, Any]) -> Dict[str, Any]:
    receipt_path = output / 'continual_stage.json'
    manifest_path = output / 'fusioneval_manifest.json'
    if not manifest_path.is_file():
        raise ExecutionError('stage output is incomplete and cannot be resumed: {}'.format(output))
    manifest = _read_json_object(manifest_path, 'stage manifest')
    expected_provenance = {'protocol': _STAGE_PROTOCOL, 'stage_plan': dict(stage_plan)}
    if manifest.get('status') != 'PASS' or manifest.get('orchestration') != expected_provenance:
        raise ExecutionError('stage atomic provenance mismatch at {}'.format(output))
    files = manifest.get('weight_files')
    if not isinstance(files, Mapping) or not files:
        raise ExecutionError('stage manifest has no weight inventory: {}'.format(output))
    for name, metadata in files.items():
        path = output / str(name)
        expected_bytes = metadata.get('bytes') if isinstance(metadata, Mapping) else None
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ExecutionError('stage weight inventory mismatch at {}'.format(path))
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, Mapping):
        raise ExecutionError('stage manifest has invalid artifact inventory: {}'.format(output))
    for name, metadata in artifacts.items():
        path = output / str(name)
        if not isinstance(metadata, Mapping):
            raise ExecutionError('stage artifact inventory is invalid at {}'.format(path))
        if metadata.get('type') == 'directory':
            inventory = metadata.get('files')
            if not path.is_dir() or not isinstance(inventory, Mapping) or (not inventory):
                raise ExecutionError('stage artifact directory is incomplete at {}'.format(path))
            for relative, item in inventory.items():
                artifact = path / str(relative)
                expected_bytes = item.get('bytes') if isinstance(item, Mapping) else None
                if not artifact.is_file() or artifact.stat().st_size != expected_bytes:
                    raise ExecutionError('stage artifact inventory mismatch at {}'.format(artifact))
        else:
            expected_bytes = metadata.get('bytes')
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ExecutionError('stage artifact inventory mismatch at {}'.format(path))
    expected_receipt = _receipt_from_manifest(output, stage_plan, manifest)
    receipt = _read_json_object(receipt_path, 'stage receipt') if receipt_path.is_file() else expected_receipt
    if receipt != expected_receipt:
        raise ExecutionError('stage receipt does not match atomic provenance at {}'.format(output))
    if not receipt_path.is_file():
        _atomic_json(receipt_path, receipt)
    return receipt

def _existing_prefix(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stages = plan.get('stages')
    if not isinstance(stages, list):
        raise ExecutionError('existing continual plan has invalid stages')
    receipts = []
    missing_seen = False
    for stage_plan in stages:
        output = Path(stage_plan['output'])
        if not output.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise ExecutionError('existing stage outputs are not a contiguous prefix')
        receipts.append(_load_stage_receipt(output, stage_plan))
    return receipts

def _same_fixed_contract(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    mutable = {'order', 'stages', 'config', 'common_config', 'effective_expert_weights'}
    return {key: value for key, value in left.items() if key not in mutable} == {key: value for key, value in right.items() if key not in mutable}

def _stage_rng_runtime(config: ContinualConfig, previous_payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config.method not in _RNG_METHODS:
        return {}
    if previous_payload is None:
        return {'rng_state': None}
    return {'rng_state': previous_payload['state_hex']}

def _stage_rng_payload(config: ContinualConfig, output: Path, stage: int, input_state: Optional[str]) -> Mapping[str, Any]:
    payload = read_stage_rng(output / 'fusioneval_manifest.json', method='continual_{}'.format(config.method), expected_device=expected_generator_device('continual_{}'.format(config.method), config.runtime['device']), stage=stage)
    if int(payload.get('initial_seed', -1)) != int(config.runtime['seed']):
        raise ExecutionError('stage {} RNG state was seeded with {} instead of the configured {}'.format(stage, payload.get('initial_seed'), config.runtime['seed']))
    expected_input = sha256_of_state(input_state) if input_state is not None else None
    if payload.get('input_state_sha256') != expected_input:
        raise ExecutionError('stage {} consumed a different random stream than the one this run supplied; the stage config and the manifest disagree'.format(stage))
    if bool(payload.get('seeded_fresh')) != (input_state is None):
        raise ExecutionError('stage {} RNG provenance is inconsistent: seeded_fresh={} with input_state={}'.format(stage, payload.get('seeded_fresh'), input_state is not None))
    if input_state is not None and payload['state_hex'] == input_state:
        raise ExecutionError('stage {} did not advance the random stream; a stage must consume draws rather than reuse the incoming state'.format(stage))
    return payload

def _run_initial_stage(config: ContinualConfig, plan: Mapping[str, Any], config_root: Path) -> Dict[str, Any]:
    stage_plan = _initial_stage_plan(plan)
    output = Path(stage_plan['output'])
    common_value, run_value = _generated_initial_configs(config, plan['initial'])
    common_path = config_root / 's00.common.yaml'
    run_path = config_root / 's00.run.yaml'
    _write_stage_yaml(common_path, common_value, completed=output.exists())
    _write_stage_yaml(run_path, run_value, completed=output.exists())
    if output.exists():
        receipt = _load_stage_receipt(output, stage_plan)
        disposition = 'REUSED'
    else:
        stage_manifest = run_method(load_config(run_path), provenance={'protocol': _STAGE_PROTOCOL, 'stage_plan': dict(stage_plan)})
        receipt = _receipt_from_manifest(output, stage_plan, stage_manifest)
        _atomic_json(output / 'continual_stage.json', receipt)
        disposition = 'CREATED'
    return {**receipt, 'disposition': disposition}

def run_continual(config: ContinualConfig, *, resume: bool=False, stop_after: Optional[int]=None) -> Dict[str, Any]:
    if config.method in _PLAN_ONLY_METHODS:
        raise ConfigError('continual {} is plan-only: runner and atomic RNG/resume support are not implemented; use --dry-run (no output directories were created)'.format(config.method))
    plan = compile_continual_plan(config)
    run_root = config.run_root
    plan_path = run_root / 'continual_plan.json'
    manifest_path = run_root / 'continual_manifest.json'
    stage_count = len(plan['stages'])
    if stop_after is not None and (isinstance(stop_after, bool) or not 1 <= stop_after <= stage_count):
        raise ConfigError('stop_after must be between 1 and {}'.format(stage_count))
    run_exists = run_root.exists()
    if resume and (not run_exists):
        raise ExecutionError('continual sequence does not exist and cannot be resumed: {}'.format(run_root))
    if run_exists and (not resume):
        raise ExecutionError('continual sequence already exists; use --resume: {}'.format(run_root))
    run_root.mkdir(parents=True, exist_ok=True)
    existing_receipts: List[Dict[str, Any]] = []
    completed = 0
    order_changed = False
    previous_order = None
    if plan_path.is_file():
        existing_plan = _read_json_object(plan_path, 'continual plan')
        existing_receipts = _existing_prefix(existing_plan)
        completed = len(existing_receipts)
        if existing_plan != plan:
            if not _same_fixed_contract(existing_plan, plan):
                raise ExecutionError('existing continual fixed contract differs')
            if existing_plan['stages'][:completed] != plan['stages'][:completed]:
                raise ExecutionError('completed continual prefix cannot be changed')
            previous_order = existing_plan.get('order')
            order_changed = previous_order != plan['order']
            _atomic_json(plan_path, plan)
    elif any(run_root.iterdir()):
        raise ExecutionError('continual sequence contains files but no frozen plan: {}'.format(run_root))
    else:
        _atomic_json(plan_path, plan)
    previous_state = _read_json_object(manifest_path, 'sequence manifest') if resume and manifest_path.is_file() else {}
    if stop_after is not None and stop_after < len(existing_receipts):
        raise ExecutionError('stop_after cannot move a sequence backward from stage {}'.format(len(existing_receipts)))
    started_at = previous_state.get('started_at_utc', datetime.now(timezone.utc).isoformat())
    receipts: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {'schema_version': 1, 'status': 'RUNNING', 'numbering': plan['numbering'], 'observation_points': plan['observation_points'], 'sequence_output': str(run_root), 'plan': str(plan_path), 'method': config.method, 'order': list(config.order), 'initial': None, 'started_at_utc': started_at, 'resume_count': int(previous_state.get('resume_count', 0)) + int(resume), 'order_changed_on_resume': order_changed, 'completed_stages': receipts}
    if order_changed:
        state['previous_order'] = previous_order
    _atomic_json(manifest_path, state)
    active_stage = None
    try:
        config_root = run_root / 'configs'
        if plan['initial']['executes']:
            initial_receipt = _run_initial_stage(config, plan, config_root)
        else:
            initial_receipt = {**_write_pointer_stage0(plan), 'disposition': 'REFERENCED'}
        state['initial'] = initial_receipt
        _atomic_json(manifest_path, state)
        rng_previous: Optional[Mapping[str, Any]] = None
        for stage_plan in plan['stages'][:stop_after]:
            stage = int(stage_plan['stage'])
            active_stage = stage
            output = Path(stage_plan['output'])
            runtime_extra = _stage_rng_runtime(config, rng_previous)
            common_value, run_value = _generated_stage_configs(config, stage_plan, runtime_extra)
            common_path = config_root / 's{:02d}.common.yaml'.format(stage)
            run_path = config_root / 's{:02d}.run.yaml'.format(stage)
            _write_stage_yaml(common_path, common_value, completed=stage <= completed)
            _write_stage_yaml(run_path, run_value, completed=stage <= completed)
            if output.exists():
                receipt = _load_stage_receipt(output, stage_plan)
                disposition = 'REUSED'
            else:
                stage_manifest = run_method(load_config(run_path), provenance={'protocol': _STAGE_PROTOCOL, 'stage_plan': dict(stage_plan)})
                receipt = _receipt_from_manifest(output, stage_plan, stage_manifest)
                _atomic_json(output / 'continual_stage.json', receipt)
                disposition = 'CREATED'
            if config.method in _RNG_METHODS:
                rng_previous = _stage_rng_payload(config, output, stage, runtime_extra.get('rng_state'))
            receipts.append({**receipt, 'disposition': disposition, **({'rng_state_sha256': rng_previous['state_sha256']} if config.method in _RNG_METHODS else {})})
            state['completed_stages'] = receipts
            _atomic_json(manifest_path, state)
    except Exception as exc:
        state['status'] = 'FAILED'
        state['failed_at_utc'] = datetime.now(timezone.utc).isoformat()
        state['failure'] = '{}: {}'.format(type(exc).__name__, exc)
        state['failed_stage'] = active_stage
        _atomic_json(manifest_path, state)
        raise
    executed = list(receipts)
    if plan['initial']['executes']:
        executed = [initial_receipt] + executed
    if stop_after is not None and stop_after < stage_count:
        state['status'] = 'PAUSED'
        state['paused_at_utc'] = datetime.now(timezone.utc).isoformat()
        state['next_stage'] = stop_after + 1
    else:
        state['status'] = 'PASS'
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    state['final_checkpoint'] = receipts[-1]['output']
    state['stage_count'] = len(receipts)
    state['algorithm_runtime_seconds'] = sum((float(item['algorithm_runtime_seconds']) for item in executed))
    state['execution_runtime_seconds'] = sum((float(item['algorithm_runtime_seconds']) for item in executed if item['disposition'] == 'CREATED'))
    _atomic_json(manifest_path, state)
    return state
