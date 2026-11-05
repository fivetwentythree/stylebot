"""Rewrite a document using the trained checkpoint and its matching policy."""

import argparse
import json
import sys
from pathlib import Path

from stylebot import backends


def serving_options(manifest, prompt):
    if (manifest.get("data") or {}).get("style_system") != prompt:
        raise ValueError(
            "The policy does not match the prompt saved with this training run."
        )
    checkpoint = (manifest.get("result") or {}).get("checkpoints") or {}
    if not checkpoint.get("sampler_path"):
        raise ValueError("This run does not contain a trained sampler checkpoint.")
    return backends.opts_from_manifest(
        manifest,
        backends.BackendOpts(backend="tinker", system_prompt=prompt),
        default_system=prompt,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="Read stdin if omitted.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    prompt = policy.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        parser.error("The policy must contain a nonempty system_prompt string.")
    try:
        backend = serving_options(manifest, prompt)
    except ValueError as error:
        parser.error(str(error))
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    styler = backends.build_styler(backend)
    result = backends.rewrite_document(text, backends.RunOpts(), backend, styler)
    sys.stdout.write(result.text)


if __name__ == "__main__":
    main()
