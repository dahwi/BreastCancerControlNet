import os
import re

# Resolve environment variables embedded within strings
def resolve_env_vars(value):
    if isinstance(value, str):
        # Match all occurrences of ${VAR_NAME}
        matches = re.findall(r"\${([^}]+)}", value)
        for match in matches:
            # Replace ${VAR_NAME} with its environment variable value, or leave it unchanged if not set
            env_value = os.getenv(match, f"${{{match}}}")  # Keep as ${VAR_NAME} if not set
            value = value.replace(f"${{{match}}}", env_value)
    return value

# Recursively resolve environment variables in the configuration
def resolve_config(config):
    if isinstance(config, dict):
        return {key: resolve_config(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [resolve_config(item) for item in config]
    else:
        return resolve_env_vars(config)