import dataclasses
import logging

import tyro

from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    asset_id: str | None = None
    default_prompt: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    record: bool = False


def main(args: Args) -> None:
    policy = policy_config.create_trained_policy(
        training_config.get_config(args.config_name),
        args.checkpoint_dir,
        asset_id=args.asset_id,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata
    if args.record:
        policy = policy_module.PolicyRecorder(policy, "policy_records")
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

