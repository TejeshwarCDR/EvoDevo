"""Launch refusal checks shared by `python -m evodevo.campaign` and
scripts/campaign/run_campaign.sh (which runs them via --check-only before it
creates the tmux session, so a refusal is printed where the operator sees
it, not inside a detached window).

Each check returns human-readable refusal reasons plus a record that the
scheduler writes into the manifest (`launch_checks`).
"""
from __future__ import annotations

import os

from .alerts import NO_NOTIFIER_MESSAGE, Notifier, load_local_config, notify_config_path
from .idle_gate import check_machine_idle


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


def mirror_check(args) -> tuple:
    if args.no_mirror:
        return [], {"no_mirror": True}
    from .mirror import check_mirror_location
    return check_mirror_location(args.output_root, args.mirror_dir,
                                 allow_same_disk=bool(getattr(args, "allow_same_disk_mirror", False)))


def launch_refusals(args):
    """-> (refusals, record). Empty refusals means the launch may proceed."""
    # Keep the no-argument call shape for existing integrations while an
    # explicit CLI selection routes to the requested local config.
    local_config = getattr(args, "local_config", None)
    notifier = notifier_status(local_config) if local_config else notifier_status()
    refusals = notifier_refusals(args, notifier)
    mirror_refusals, mirror_record = mirror_check(args)
    refusals += mirror_refusals
    if args.mode == "validation":
        idle_refusals, idle_record = check_machine_idle()
        refusals += idle_refusals
    else:
        idle_record = {"required": False}
    record = {"notifier": notifier, "no_notify": bool(args.no_notify), "mode": args.mode,
              "mirror": mirror_record,
              "machine_idle": idle_record,
              # set by run_campaign.sh: "systemd-inhibit" when the sleep/idle/lid lock is held
              "sleep_inhibitor": os.environ.get("EVODEVO_SLEEP_INHIBITOR", "none")}
    return refusals, record
