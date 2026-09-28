"""Launch refusal checks shared by `python -m evodevo.campaign` and
scripts/campaign/run_campaign.sh (which runs them via --check-only before it
creates the tmux session, so a refusal is printed where the operator sees
it, not inside a detached window).

Each check returns human-readable refusal reasons plus a record that the
scheduler writes into the manifest (`launch_checks`).
"""
from __future__ import annotations

from .alerts import NO_NOTIFIER_MESSAGE, Notifier, load_local_config, notify_config_path


def notifier_status(config_path=None) -> dict:
    path = notify_config_path(config_path)
    notifier = Notifier(load_local_config(path))
    return {"config_path": path, "configured": notifier.configured, "channels": notifier.channels}


def notifier_refusals(args, status: dict) -> list:
    if args.mode != "real" or status["configured"] or args.no_notify:
        return []
    return [f"{NO_NOTIFIER_MESSAGE}: {status['config_path']} configures neither ntfy_topic_url nor "
            "shell_command. Configure one (and prove delivery with `python -m evodevo.campaign.alerts "
            "--test`), or pass --no-notify explicitly."]


def launch_refusals(args):
    """-> (refusals, record). Empty refusals means the launch may proceed."""
    notifier = notifier_status()
    refusals = notifier_refusals(args, notifier)
    record = {"notifier": notifier, "no_notify": bool(args.no_notify), "mode": args.mode}
    return refusals, record
