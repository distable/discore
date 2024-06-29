import json
import sys
import traceback
import urllib
import uuid
from pathlib import Path

import PIL
import numpy as np
import websocket
from PIL.Image import Image

import userconf
from src.classes.Plugin import Plugin
from src_plugins.comfy import comfy_serverless
from src_plugins.comfy.comfy_serverless import ComfyConnector

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

image_input_path = 'D:\Projects\ComfyUI_windows_portable\ComfyUI\input'
created_input_images = []


class ComfyPlugin(Plugin):
    def title(self):
        return "comfy"

    def describe(self):
        return ""

    def txt2img(self):
        pass

    def img2img(self):
        pass

    def workflow(self, name='workflow', **inputs):
        from src import renderer
        rv = renderer.rv

        def try_path(path):
            wf_path = (Path(path) / f'{name}.json')
            api_path = (Path(path) / f'{name}_api.json')
            if not wf_path.exists():
                return False, None, None

            wf_json_text = wf_path.read_text()
            api_json_text = api_path.read_text()
            return success, api_json_text, wf_json_text

        success, api_json_text, wf_json_text = try_path(name)
        if not success:
            success, api_json_text, wf_json_text = try_path(rv.session.dirpath)

        workflow = json.loads(wf_json_text)
        prompt = json.loads(api_json_text)

        # First we test the node titles to see if we have dedicated inputs for them
        for node in workflow['nodes']:
            id = node['id']
            type = node['type']
            title = node.get('title', '')
            if title not in inputs:
                print(f"Skipping {title} because it is not in inputs")
                continue

            if f"{id}" not in prompt:
                # # Match by value
                # for pnode in prompt.values():
                #     for pnode_input in pnode.inputs.items():
                print(f"Skipping {title} because it is not in prompt")
                continue

            pnode = prompt[f"{id}"]
            value = inputs[title]
            if type == 'LoadImage':
                image_dir = Path(image_input_path)
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f'{title}.png'

                if isinstance(value, Image):
                    value.save(image_path)
                elif isinstance(value, np.ndarray):
                    PIL.Image.fromarray(value).save(image_path)

                created_input_images.append(image_path)
                ComfyConnector.replace_key_value(pnode, 'image', image_path.name)
            elif type in ['CLIPTextEncode', 'DPRandomGenerator']:
                ComfyConnector.replace_key_value(pnode, 'text', value)
            elif type in ['String Literal']:
                ComfyConnector.replace_key_value(pnode, 'string', value)
            elif type in ['Float', 'Int', 'Int Literal']:
                ComfyConnector.replace_key_value(pnode, 'Value', value)
                ComfyConnector.replace_key_value(pnode, 'int', value)
                ComfyConnector.replace_key_value(pnode, 'float', value)
                ComfyConnector.replace_key_value(pnode, 'number', value)
            elif type in ['CLIPSetLastLayer']:
                ComfyConnector.replace_key_value(pnode, 'stop_at_clip_layer', value)
            else:
                print(f"Warning: {type} not handled")
                ComfyConnector.replace_key_value(pnode, title, value)
                ComfyConnector.replace_key_value(pnode, 'value', value)
                ComfyConnector.replace_key_value(pnode, 'float', value)
            inputs.pop(title)

        # Otherwise the rest of the inputs are passed to the first parameter that matches
        for key, value in inputs.items():
            ComfyConnector.replace_key_value(prompt, key, value)

        connector = ComfyConnector()
        images = connector.generate_images(prompt)

        # for image in created_input_images:
        #     image.unlink()
        created_input_images.clear()

        return images[0]

