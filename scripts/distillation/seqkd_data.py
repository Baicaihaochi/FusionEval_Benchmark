import json
from pathlib import Path


def training_view(row):
    md = row['metadata']
    tokens = md['seqkd_token_ids']
    length = md['seqkd_response_length']
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= 4096:
        raise ValueError('Expected an exact prefix of 2..4096 tokens')
    if not 0 <= length < len(tokens):
        raise ValueError('Response length must fit within the token sequence')
    if length:
        return tokens, length, [1] * length
    return tokens, 1, [0]


class SeqKDDataSource:
    def __init__(self, args):
        self.args = args
        self.path = Path(args.prompt_data)
        self.offsets = []
        with self.path.open('rb') as handle:
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                self.offsets.append(pos)
        self.cursor = 0

    def __len__(self):
        return len(self.offsets)

    def get_samples(self, num_samples):
        from slime.utils.types import Sample
        if self.cursor + num_samples > len(self.offsets):
            raise ValueError('Not enough samples for the next batch')
        groups = []
        with self.path.open('rb') as handle:
            handle.seek(self.offsets[self.cursor])
            for index in range(self.cursor, self.cursor + num_samples):
                row = json.loads(handle.readline())
                tokens, length, mask = training_view(row)
                sample = Sample(
                    index=index, group_index=index, rollout_id=index,
                    prompt='', tokens=tokens, response='', response_length=length,
                    reward=0, metadata=row['metadata'], loss_mask=mask,
                )
                sample.status = Sample.Status.COMPLETED
                groups.append([sample])
        self.cursor += num_samples
        return groups

    def add_samples(self, samples):
        raise RuntimeError('This dataset is fixed-order and cannot accept samples')

    def save(self, rollout_id):
        root = Path(self.args.save) / 'rollout'
        root.mkdir(exist_ok=True, parents=True)
        (root / f'seqkd_cursor_{rollout_id}.json').write_text(json.dumps({'cursor': self.cursor}) + '\n')

    def load(self, rollout_id=None):
        if rollout_id is None or rollout_id < 0:
            return
        p = Path(self.args.load) / 'rollout' / f'seqkd_cursor_{rollout_id}.json'
        self.cursor = json.loads(p.read_text())['cursor']


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    return data_buffer.get_samples(args.rollout_batch_size)
