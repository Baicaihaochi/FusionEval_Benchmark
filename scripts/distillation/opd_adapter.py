from copy import copy

from slime.rollout import on_policy_distillation as opd
from slime.rollout.multi_teacher_on_policy_distillation import parse_teacher_urls, select_teacher


async def reward_func(args, sample, **kwargs):
    teacher, url = select_teacher(
        sample, parse_teacher_urls(args.opd_teacher_urls), args.opd_teacher_metadata_key
    )
    routed = copy(args)
    routed.rm_url = url
    reward = await opd.reward_func(routed, sample, **kwargs)
    reward.setdefault('meta_info', {})['opd_teacher'] = teacher
    return reward


post_process_rewards = opd.post_process_rewards
