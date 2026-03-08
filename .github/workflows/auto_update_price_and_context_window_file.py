import asyncio
import aiohttp
import json
import re

# Asynchronously fetch data from a given URL
async def fetch_data(url, result_key='data'):
    try:
        # Create an asynchronous session
        async with aiohttp.ClientSession() as session:
            # Send a GET request to the URL
            async with session.get(url) as resp:
                # Raise an error if the response status is not OK
                resp.raise_for_status()
                # Parse the response JSON
                resp_json = await resp.json()
                print(f"Fetched data from {url}")
                if result_key:
                    return resp_json[result_key]
                return resp_json
    except Exception as e:
        # Print an error message if fetching data fails
        print("Error fetching data from URL:", e)
        return None

# Synchronize local data with remote data
def sync_local_data_with_remote(local_data, remote_data):
    # Update existing keys: merge remote fields into local, preserving any
    # manually set fields that the remote API doesn't return (e.g.
    # supports_assistant_prefill, tool_use_system_prompt_tokens, etc.)
    for key in (set(local_data) & set(remote_data)):
        local_data[key].update(remote_data[key])

    # Add new keys from remote_data to local_data
    for key in (set(remote_data) - set(local_data)):
        local_data[key] = remote_data[key]

# Write data to the json file
def write_to_file(file_path, data):
    try:
        # Open the file in write mode
        with open(file_path, "w") as file:
            # Dump the data as JSON into the file
            json.dump(data, file, indent=4)
        print("Values updated successfully.")
    except Exception as e:
        # Print an error message if writing to file fails
        print("Error updating JSON file:", e)

# Update the existing models and add the missing models for OpenRouter
def transform_openrouter_data(data):
    transformed = {}
    for row in data:
        # Add the fields 'max_tokens' and 'input_cost_per_token'
        obj = {
            "max_tokens": row["context_length"],
            "input_cost_per_token": float(row["pricing"]["prompt"]),
        }

        # Add 'max_output_tokens' as a field if it is not None
        if "top_provider" in row and "max_completion_tokens" in row["top_provider"] and row["top_provider"]["max_completion_tokens"] is not None:
            obj['max_output_tokens'] = int(row["top_provider"]["max_completion_tokens"])

        # Add the field 'output_cost_per_token'
        obj.update({
            "output_cost_per_token": float(row["pricing"]["completion"]),
        })

        # Add field 'input_cost_per_image' if it exists and is non-zero
        if "pricing" in row and "image" in row["pricing"] and float(row["pricing"]["image"]) != 0.0:
            obj['input_cost_per_image'] = float(row["pricing"]["image"])

        # Add the fields 'litellm_provider' and 'mode'
        obj.update({
            "litellm_provider": "openrouter",
            "mode": "chat"
        })

        # Add the 'supports_vision' field if image is in input modalities
        input_modalities = row.get('architecture', {}).get('input_modalities', [])
        if 'image' in input_modalities:
            obj['supports_vision'] = True

        if 'audio' in input_modalities:
            obj['supports_audio_input'] = True

        # Use a composite key to store the transformed object
        transformed[f'openrouter/{row["id"]}'] = obj

    return transformed


# Detect mode for NVIDIA NIM models based on name patterns
_NIM_EMBED_PATTERNS = ['embed', 'embedqa', 'bge-m3', 'arctic-embed', 'nv-embed', 'nemoretriever', 'embedcode', 'nvclip']
_NIM_RERANK_PATTERNS = ['rerank', 'rerankqa']
_NIM_VLM_PATTERNS = ['vision', 'vl-', '-vl-', '-vl/', 'vila', 'paligemma', 'deplot', 'kosmos', 'multimodal', 'fuyu', 'neva']
_NIM_REASONING_PATTERNS = ['thinking', 'qwq', 'flash-reasoning', 'magistral']


def _nim_detect_mode(model_id):
    mid = model_id.lower()
    for p in _NIM_RERANK_PATTERNS:
        if p in mid:
            return 'rerank'
    for p in _NIM_EMBED_PATTERNS:
        if p in mid:
            return 'embedding'
    return 'chat'


def _nim_detect_vision(model_id):
    mid = model_id.lower()
    return any(p in mid for p in _NIM_VLM_PATTERNS)


def _nim_detect_reasoning(model_id):
    mid = model_id.lower()
    if '/deepseek-r1' in mid or '-r1-' in mid or mid.endswith('-r1'):
        return True
    return any(p in mid for p in _NIM_REASONING_PATTERNS)


def _nim_detect_context(model_id):
    m = re.search(r'[-_](\d+)k[-_]', model_id.lower())
    if m:
        return int(m.group(1)) * 1024
    return None


# Update the existing models and add missing models for NVIDIA NIM
def transform_nvidia_nim_data(data):
    transformed = {}
    seen = set()
    for row in data:
        mid = row['id']
        if mid in seen:
            continue
        seen.add(mid)

        mode = _nim_detect_mode(mid)
        obj = {
            'litellm_provider': 'nvidia_nim',
            'mode': mode,
            'input_cost_per_token': 0.0,
        }

        if mode == 'chat':
            obj['output_cost_per_token'] = 0.0
            ctx = _nim_detect_context(mid)
            if ctx:
                obj['max_input_tokens'] = ctx
                obj['max_tokens'] = ctx
            if _nim_detect_vision(mid):
                obj['supports_vision'] = True
            if _nim_detect_reasoning(mid):
                obj['supports_reasoning'] = True
            instruct_families = ['instruct', 'chat', 'llama', 'mistral', 'mixtral',
                                  'nemotron', 'qwen', 'deepseek', 'gemma', 'phi',
                                  'granite', 'falcon', 'codellama', 'starcoder']
            if any(x in mid.lower() for x in instruct_families):
                obj['supports_function_calling'] = True
                obj['supports_tool_choice'] = True
        elif mode == 'rerank':
            obj['input_cost_per_query'] = 0.0
            obj['output_cost_per_token'] = 0.0
        # embedding: input_cost_per_token only (already set above)

        transformed[f'nvidia_nim/{mid}'] = obj

    return transformed

# Update the existing models and add the missing models for Vercel AI Gateway
def transform_vercel_ai_gateway_data(data):
    transformed = {}
    for row in data:
        pricing = row.get("pricing", {})
        obj = {
            "max_tokens": row["context_window"],
            "input_cost_per_token": float(pricing.get("input", 0) or 0),
            "output_cost_per_token": float(pricing.get("output", 0) or 0),
            'max_output_tokens': row.get('max_tokens', row["context_window"]),
            'max_input_tokens': row["context_window"],
        }

        # Handle cache pricing if available
        if pricing.get("input_cache_read") is not None:
            obj['cache_read_input_token_cost'] = float(f"{float(pricing['input_cache_read']):e}")

        if pricing.get("input_cache_write") is not None:
            obj['cache_creation_input_token_cost'] = float(f"{float(pricing['input_cache_write']):e}")

        mode = "embedding" if "embedding" in row["id"].lower() else "chat"

        obj.update({"litellm_provider": "vercel_ai_gateway", "mode": mode})

        transformed[f'vercel_ai_gateway/{row["id"]}'] = obj

    return transformed


# Load local data from a specified file
def load_local_data(file_path):
    try:
        # Open the file in read mode
        with open(file_path, "r") as file:
            # Load and return the JSON data
            return json.load(file)
    except FileNotFoundError:
        # Print an error message if the file is not found
        print("File not found:", file_path)
        return None
    except json.JSONDecodeError as e:
        # Print an error message if JSON decoding fails
        print("Error decoding JSON:", e)
        return None

def main():
    local_file_path = "model_prices_and_context_window.json"  # Path to the local data file
    openrouter_url = "https://openrouter.ai/api/v1/models"
    vercel_ai_gateway_url = "https://ai-gateway.vercel.sh/v1/models"
    nvidia_nim_url = "https://integrate.api.nvidia.com/v1/models"

    # Load local data from file
    local_data = load_local_data(local_file_path)

    all_remote_data = {}

    # Fetch and transform OpenRouter data
    openrouter_raw = asyncio.run(fetch_data(openrouter_url, result_key='data'))
    if openrouter_raw:
        all_remote_data.update(transform_openrouter_data(openrouter_raw))
        print(f"OpenRouter: {len(openrouter_raw)} models")
    else:
        print("WARNING: Failed to fetch OpenRouter data")

    # Fetch and transform Vercel AI Gateway data
    vercel_raw = asyncio.run(fetch_data(vercel_ai_gateway_url, result_key='data'))
    if vercel_raw:
        all_remote_data.update(transform_vercel_ai_gateway_data(vercel_raw))
        print(f"Vercel AI Gateway: {len(vercel_raw)} models")
    else:
        print("WARNING: Failed to fetch Vercel AI Gateway data")

    # Fetch and transform NVIDIA NIM data
    nvidia_nim_raw = asyncio.run(fetch_data(nvidia_nim_url, result_key='data'))
    if nvidia_nim_raw:
        all_remote_data.update(transform_nvidia_nim_data(nvidia_nim_raw))
        print(f"NVIDIA NIM: {len(nvidia_nim_raw)} models")
    else:
        print("WARNING: Failed to fetch NVIDIA NIM data")

    print(f"Total remote entries to sync: {len(all_remote_data)}")

    if local_data and all_remote_data:
        sync_local_data_with_remote(local_data, all_remote_data)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    main()
