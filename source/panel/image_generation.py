import re
import json
import logging
import os
import time
from datetime import datetime
from typing import List
import boto3
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from tenacity import stop_after_attempt, retry

logging.basicConfig(level=logging.INFO)
# logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# load .env file with specific name
load_dotenv(dotenv_path='.env_sd')

# Your ApiGatewayUrl in Extension for Stable Diffusion
# Example: https://xxxx.execute-api.us-west-2.amazonaws.com/prod/
COMMAND_API_URL = os.getenv("COMMON_API_URL")
# Your ApiGatewayUrlToken in Extension for Stable Diffusion
API_KEY = os.getenv("API_KEY")
# Your username in Extension for Stable Diffusion
# Some resources are limited to specific users
API_USERNAME = os.getenv("API_USERNAME")
# The service support varies in different regions
BEDROCK_REGION = os.getenv("BEDROCK_REGION")

# API URL
GENERATE_API_URL = COMMAND_API_URL + "inference/v2"
STATUS_API_URL = COMMAND_API_URL + "inference/get-inference-job"
PARAM_API_URL = COMMAND_API_URL + "inference/get-inference-job-param-output"
IMAGE_API_URL = COMMAND_API_URL + "inference/get-inference-job-image-output"
CHECKPOINTS_API_URL = COMMAND_API_URL + "checkpoints"

support_model_list = []

# fixed model list installed on sd solution, MUST align with prompt template
prompt_model_list = [
    {"sd_xl_base_1.0.safetensors", "default"},
    {"majicmixRealistic_v7.safetensors", "realistic"},
    {"x2AnimeFinal_gzku.safetensors", "anime"},
    {"LahCuteCartoonSDXL_alpha.safetensors", "cartoon"}
]

default_models = ["sd_xl_base_1.0.safetensors"]

# todo will update api
def deploy_sagemaker_endpoint(instance_type: str = "ml.g4dn.4xlarge", initial_instance_count: int = 1,
                              endpoint_name: str = "default-endpoint-for-llm-bot"):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': API_KEY
    }
    inputBody = {
        "instance_type": instance_type,
        "initial_instance_count": initial_instance_count,
        "endpoint_name": endpoint_name
    }
    # https://<Your API Gateway ID>.execute-api.<Your AWS Account Region>.amazonaws.com/{basePath}/inference/deploy-sagemaker-endpoint
    res = requests.post(COMMAND_API_URL + 'inference/deploy-sagemaker-endpoint', headers=headers, json=inputBody)
    logger.info("deploy_sagemaker_endpoint: {}".format(res.json()))

def upload_model():
    pass

def get_bedrock_llm():
    # specify the profile_name to call the bedrock api if needed
    bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)

    modelId = "anthropic.claude-v2"
    cl_llm = Bedrock(
        model_id=modelId,
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 1000},
    )
    return cl_llm

# todo template use dynamic checkpoints
# sd_prompt = PromptTemplate.from_template(
#     """
#     Human:
#     - Transform the input prompt {input} into a detailed prompt for an image generation model, describing the scene with vivid and specific attributes that enhance the original concept, only adjective and noun are allowed, verb and adverb are not allowed, each words speperated by comma.
#     - Generate a negative prompt that specifies what should be avoided in the image, including any elements that contradict the desired style or tone.
#     - Recommend a list of suitable models from the fix model list that best match the style and content described in the detailed prompt.
#     - Other notes please refer to the following example:

#     The output should be a plain text in Python List format shown follows, no extra content added beside Positive Prompt, Negative Prompt and Recommended Model Index List.
#     [Positive Prompt: <detailed_prompt>,
#     Negative Prompt: <negative_prompt>,
#     Recommended Model Index List: [model index list]]

#     The model can only be chosen from following list of dict with table name and style name described, we can only and must choose 2 models based on style described and output the model index list:
#     [
#         {{"sd_xl_base_1.0.safetensors", "default"}},
#         {{"majicmixRealistic_v7.safetensors", "realistic"}},
#         {{"x2AnimeFinal_gzku.safetensors", "anime"}}
#         {{"LahCuteCartoonSDXL_alpha.safetensors", "cartoon"}}
#     ]

#     For example:
#     If the input prompt is: "a cute dog in cartoon style", the output should be as follows:
#     [
#         Positive Prompt: "visually appealing, high-quality image of a cute dog in a vibrant, cartoon style, adorable appearance, expressive eyes, friendly demeanor, colorful and lively, reminiscent of popular animation studios, artwork.",
#         Negative Prompt: "realism, dark or dull colors, scary or aggressive dog depictions, overly simplistic, stick figure drawings, blurry or distorted images, inappropriate or NSFW content.",
#         Recommended Model Index List: [0, 3]
#     ]

#     If the input prompt is: "a girl in photo-realistic style", the output should be as follows:
#     [
#         Positive Prompt: "detailed, photo-realistic, life-like, high-definition, sharp, accurate color tones, realistic textures, natural lighting, subtle expressions, vivid, true-to-life, authentic appearance, nuanced, real photograph.",
#         Negative Prompt: "cartoonish, abstract, stylized, overly simplistic, exaggerated, distorted features, bright unrealistic colors, artificial elements, fantasy elements, non-photo-realistic.",
#         Recommended Model Index List: [0, 2]
#     ]

#     Current conversation:
#     <conversation_history>
#     {history}
#     </conversation_history>

#     Here is the human's next reply:
#     <human_reply>
#     {input}
#     </human_reply>

#     Assistant:
#     """)

sd_prompt = PromptTemplate.from_template(
    """
    Human:
    - Transform the input prompt {input} into a detailed prompt for an image generation model, describing the scene with vivid and specific attributes that enhance the original concept, only adjective and noun are allowed, verb and adverb are not allowed, each words speperated by comma.
    - Generate a negative prompt that specifies what should be avoided in the image, including any elements that contradict the desired style or tone.
    - Other notes please refer to the following example:

    The output should be a plain text in Python List format shown follows, no extra content added beside Positive Prompt, Negative Prompt.
    [Positive Prompt: <detailed_prompt>,
    Negative Prompt: <negative_prompt>]

    For example:
    If the input prompt is: "a cute dog in cartoon style", the output should be as follows:
    [
        Positive Prompt: "visually appealing, high-quality image of a cute dog in a vibrant, cartoon style, adorable appearance, expressive eyes, friendly demeanor, colorful and lively, reminiscent of popular animation studios, artwork.",
        Negative Prompt: "realism, dark or dull colors, scary or aggressive dog depictions, overly simplistic, stick figure drawings, blurry or distorted images, inappropriate or NSFW content.",
    ]

    If the input prompt is: "a girl in photo-realistic style", the output should be as follows:
    [
        Positive Prompt: "detailed, photo-realistic, life-like, high-definition, sharp, accurate color tones, realistic textures, natural lighting, subtle expressions, vivid, true-to-life, authentic appearance, nuanced, real photograph.",
        Negative Prompt: "cartoonish, abstract, stylized, overly simplistic, exaggerated, distorted features, bright unrealistic colors, artificial elements, fantasy elements, non-photo-realistic.",
    ]

    If the input prompt is: "Can you draw a photograph of astronaut floating in space", the output should be as follows:
    [
        Positive Prompt: "breathtaking selfie photograph of astronaut floating in space, earth in the background. masterpiece, best quality, highly detailed",
        Negative Prompt: "lowres, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry, cropped, worst quality, low quality, watermark",
    ]

    If the input prompt is: "帮我画一个飞行的宇航员", the output should be as follows:
    [
        Positive Prompt: "breathtaking selfie photograph of astronaut floating in space, earth in the background. masterpiece, best quality, highly detailed",
        Negative Prompt: "lowres, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry",
    ]

    If the input prompt is: "Please generate a night street of Tokyo", the output should be as follows:
    [
        Positive Prompt: "breathtaking night street of Tokyo, neon lights. masterpiece, best quality, highly detailed",
        Negative Prompt: "lowres, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry",
    ]

    If the input prompt is: "你可以画一个东京的夜景图吗？", the output should be as follows:
    [
        Positive Prompt: "breathtaking night street of Tokyo, neon lights. masterpiece, best quality, highly detailed",
        Negative Prompt: "lowres, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry",
    ]

    If the input prompt is: "Generate an empty classroom in anime style", the output should be as follows:
    [
        Positive Prompt: "anime artwork an empty classroom. anime style, key visual, vibrant, studio anime, highly detailed",
        Negative Prompt: "photo, deformed, black and white, realism, disfigured, low contrast",
    ]

    If the input prompt is: "i want a photo of 1 realistic girl", the output should be as follows:
    [
        Positive Prompt: "masterpiece, best quality,realistic,1girl",
        Negative Prompt: "nsfw,(worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), watermark, (bad-hands-5:1.5)",
    ]

    If the input prompt is: "我想要一张女生的写实照片", the output should be as follows:
    [
        Positive Prompt: "best quality,highly detailed, masterpiece, 8k wallpaper, realistic,1girl",
        Negative Prompt: "nsfw,(worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), watermark, (bad-hands-5:1.5)",
    ]

    If the input prompt is: "我想要一张漫画风格的小狗", the output should be as follows:
    [
        Positive Prompt: "masterpiece, best quality, cartoon style, dog",
        Negative Prompt: "realistic, dark or dull colors, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), watermark",
    ]

    If the input prompt is: "请帮忙画一张朋克风格的猫", the output should be as follows:
    [
        Positive Prompt: "masterpiece, best quality, cybernetic cat wears futuristic armor",
        Negative Prompt: "dark environment, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), watermark",
    ]
    
    If the input prompt is: "please draw a Steampunk cat", the output should be as follows:
    [
        Positive Prompt: "masterpiece, best quality, cybernetic cat wears futuristic armor",
        Negative Prompt: "dark environment, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), watermark",
    ]
 
    "Do not include style modifiers that do not appear in the question in positive prompt, such as cute, cartoon \n"   

    Current conversation:
    <conversation_history>
    {history}
    </conversation_history>

    Here is the human's next reply:
    <human_reply>
    {input}
    </human_reply>

    Assistant:
    """)

sd_prompt_cn = PromptTemplate.from_template(
    """
    Human:
    - 将输入提示 {input} 转化为图像生成模型的详细提示，用生动和具体的属性描述场景，以增强原始概念，只允许使用形容词和名词，不允许使用动词和副词，每个词用逗号分隔。
    - 生成一个负面提示，指定图像中应避免的内容，包括任何与期望风格或基调相矛盾的元素。
    - 从固定的模型列表中推荐最符合详细提示描述的风格和内容的模型列表。
    - 其他注意事项请参考以下示例：

    输出应为以下所示的 Python 列表格式纯文本，除了 Positive Prompt、Negative Prompt 和 Recommended Model Index List 之外不添加任何额外内容。模型列表只能从以下基于风格描述的列表中选择，并仅输出索引号：
    [
        {{"sd_xl_base_1.0.safetensors", "default"}},
        {{"majicmixRealistic_v7.safetensors", "realistic"}},
        {{"x2AnimeFinal_gzku.safetensors", "anime"}}
        {{"LahCuteCartoonSDXL_alpha.safetensors", "catoon"}}
    ]

    [Positive Prompt: <detailed_prompt>,
    Negative Prompt: <negative_prompt>,
    Recommended Model Index List: <model index list>]

    例如：
    如果输入提示是：“卡通风格的可爱狗”，输出应如下：
    [
        Positive Prompt: "visually appealing, high-quality image of a cute dog in a vibrant, cartoon style, adorable appearance, expressive eyes, friendly demeanor, colorful and lively, reminiscent of popular animation studios, artwork.",
        Negative Prompt: "realism, dark or dull colors, scary or aggressive dog depictions, overly simplistic, stick figure drawings, blurry or distorted images, inappropriate or NSFW content.",
        Recommended Model Index List: [0, 3]
    ]

    当前对话：
    <conversation_history>
    {history}
    </conversation_history>

    人类的下次回复：
    <human_reply>
    {input}
    </human_reply>

    Assistant:
    """)

summary_prompt = PromptTemplate.from_template(
    """
    Human:
    - According the input prompt {input} to generate summary for the image generated by the model.
    Current conversation:
    <conversation_history>
    {history}
    </conversation_history>

    Here is the human's next reply:
    <human_reply>
    {input}
    </human_reply>

    Assistant:
    """)

def get_llm_processed_prompts(initial_prompt):
    cl_llm = get_bedrock_llm()
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=cl_llm, verbose=False, memory=memory
    )

    # use cn template to avoid instability in prompt output
    conversation.prompt = sd_prompt_cn if check_if_input_cn(initial_prompt) else sd_prompt

    response = conversation.predict(input=initial_prompt)
    logger.info("the first invoke: {}".format(response))
    # logger.info("the second invoke: {}".format(conversation.predict(input="change to realist style")))

    # TODO, below paras is not stable and can be changed accord to PE, will update later
    # Define regular expressions
    positive_pattern = r"Positive Prompt: (.*?),\s+Negative Prompt:"
    negative_pattern = r"Negative Prompt: (.*?),\s+]"


    # Extract data using regex
    positive_prompt = re.search(positive_pattern, response, re.DOTALL).group(1).strip()
    negative_prompt = re.search(negative_pattern, response, re.DOTALL).group(1).strip()
    # model_index_list = re.search(model_pattern, response, re.DOTALL).group(1).strip()
    model_index_list = []
    logger.info("positive_pattern: {}".format(positive_prompt))
    logger.info("negative_pattern: {}".format(negative_prompt))

    logger.info("positive_prompt: {}\n negative_prompt: {}\n model_index_list: {}".format(positive_prompt, negative_prompt, model_index_list))
    return positive_prompt, negative_prompt, model_index_list

def get_llm_summary(initial_prompt):
    cl_llm = get_bedrock_llm()
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=cl_llm, verbose=False, memory=memory
    )
    conversation.prompt = summary_prompt
    response = conversation.predict(input=initial_prompt)
    logger.info("summary response: {}".format(response))
    return response

def generate_image(positive_prompts: str, negative_prompts: str, model: List[str], current_col, progress_bar):
    job = create_inference_job(model)
    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    inference = job["inference"]

    upload_inference_job_api_params(inference["api_params_s3_upload_url"], positive_prompts, negative_prompts)
    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    run_inference_job(inference["id"])
    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    while True:
        status_response = get_inference_job(inference["id"])
        if st.session_state.progress < 80:
            st.session_state.progress += 10
        progress_bar.progress(st.session_state.progress)
        if status_response['status'] == 'succeed':
            progress_bar.progress(100)
            image_url = get_inference_image_output(inference["id"])[0]
            current_col.image(image_url, use_column_width=True)
            break
        elif status_response['status'] == 'failed':
            current_col.error("Image generation failed.")
            break
        else:
            time.sleep(1)

    api_params = get_inference_param_output(inference["id"])
    params = requests.get(api_params[0]).json()
    info = json.loads(params['info'])

    # debug info
    if debug:
        if info["prompt"] != "":
            current_col.write("prompt:")
            current_col.info(info["prompt"])

        if info["negative_prompt"] != "":
            current_col.write("negative_prompt:")
            current_col.info(info["negative_prompt"])

        if info["sd_model_name"] != "":
            current_col.write("sd_model_name:")
            current_col.info(info["sd_model_name"])

        for warning in st.session_state.warnings:
            current_col.warning(warning)

    return inference["id"]

def get_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.get(STATUS_API_URL, headers=headers, params={"jobID": inference_id})

    return job.json()

def get_inference_param_output(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.get(PARAM_API_URL, headers=headers, params={"jobID": inference_id})

    return job.json()

def get_inference_image_output(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.get(IMAGE_API_URL, headers=headers, params={"jobID": inference_id})

    return job.json()

def get_checkpoints():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    params = {
        "username": API_USERNAME,
        "status": "Active",
    }

    job = requests.get(CHECKPOINTS_API_URL, headers=headers, params=params)
    checkpoints = []
    if 'checkpoints' in job.json():
        for checkpoint in job.json()['checkpoints']:
            checkpoints.append(checkpoint['name'][0])

    if len(checkpoints) == 0:
        raise Exception("No checkpoint available.")

    global support_model_list
    support_model_list = checkpoints
    # filter the model v1-5-pruned-emaonly.safetensors out of support list
    support_model_list = [item for item in support_model_list if not item.startswith("v1-5-pruned-emaonly")]
    logger.info("support_model_list: {}".format(support_model_list))
    return support_model_list

def create_inference_job(models: List[str]):
    models = select_checkpoint(models)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    # todo use default api params
    body = {
        "user_id": API_USERNAME,
        "task_type": "txt2img",
        "models": {
            "Stable-diffusion": models,
            "embeddings": []
        },
        "filters": {
            "createAt": datetime.now().timestamp(),
            "creator": "sd-webui"
        }
    }

    job = requests.post(GENERATE_API_URL, headers=headers, json=body)
    return job.json()

def run_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.put(COMMAND_API_URL + 'inference/v2/' + inference_id + '/run', headers=headers)

    return job.json()

def upload_inference_job_api_params(s3_url, positive: str, negative: str):
    # todo use default api params
    api_params = {
        "prompt": positive,
        "negative_prompt": negative,
        "styles": [],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0.0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "restore_faces": None,
        "tiling": None,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "eta": None,
        "denoising_strength": None,
        "s_min_uncond": 0.0,
        "s_churn": 0.0,
        "s_tmax": "Infinity",
        "s_tmin": 0.0,
        "s_noise": 1.0,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "refiner_checkpoint": None,
        "refiner_switch_at": None,
        "disable_extra_networks": False,
        "comments": {},
        "enable_hr": False,
        "firstphase_width": 0,
        "firstphase_height": 0,
        "hr_scale": 2.0,
        "hr_upscaler": "Latent",
        "hr_second_pass_steps": 0,
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_checkpoint_name": None,
        "hr_sampler_name": None,
        "hr_prompt": "",
        "hr_negative_prompt": "",
        "sampler_index": "DPM++ 2M Karras",
        "script_name": None,
        "script_args": [],
        "send_images": True,
        "save_images": False,
        "alwayson_scripts": {
            "refiner": {
                "args": [False, "", 0.8]
            },
            "seed": {
                "args": [-1, False, -1, 0, 0, 0]
            },
            "controlnet": {
                "args": [
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    },
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    },
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    }
                ]
            },
            "segment anything": {
                "args": [
                    False,
                    False,
                    0,
                    None,
                    [],
                    0,
                    False,
                    [],
                    [],
                    False,
                    0,
                    1,
                    False,
                    False,
                    0,
                    None,
                    [],
                    -2,
                    False,
                    [],
                    False,
                    0,
                    None,
                    None
                ]
            },
            "extra options": {
                "args": []
            }
        }
    }

    json_string = json.dumps(api_params)
    response = requests.put(s3_url, data=json_string)
    response.raise_for_status()
    return response


def generate_llm_image(initial_prompt: str, col, order: int):
    st.spinner()
    st.session_state.progress = 5
    progress_bar = col.progress(st.session_state.progress)

    try:
        generate_llm_image_col(initial_prompt, col, order, progress_bar)
        st.session_state.succeed_count += 1
    except Exception as e:
        # Exceed the retry limit
        col.error("Image generation failed, please try again.")
        raise e


@retry(stop=stop_after_attempt(1))
def generate_llm_image_col(initial_prompt: str, col, order: int, progress_bar):
    global support_model_list
    # models = default_models

    positive_prompt, negative_prompt, model_index_list = get_llm_processed_prompts(initial_prompt)
    st.session_state.progress += 15
    progress_bar.progress(st.session_state.progress)

    # if prompt is empty, use default
    if positive_prompt == "" or negative_prompt == "":
        positive_prompt = initial_prompt

    # if len(model_index_list) > 0:
    #     # TODO, support model list should align with prompt template, we assume the model list is fixed at 2
    #     models = [support_model_list[int(index)] for index in model_index_list.strip("[]").split(",")]

    # select the model in model list according to the order while keep the List type to compatible with genegrate_image, e.g. models: ['LahCuteCartoonSDXL_alpha.safetensors', 'majicmixRealistic_v7.safetensors']
    # models = [models[order]]

    # This is a synchronous call, will block the UI
    generate_image(positive_prompt, negative_prompt, default_models, col, progress_bar)

def select_checkpoint(user_list: List[str]):
    global support_model_list
    user_list = [item.strip() for item in user_list]
    intersection = list(set(user_list).intersection(set(support_model_list)))
    if len(intersection) == 0:
        intersection = default_models
        st.session_state.warnings.append("Use default model {}\nwhen LLM recommends {} not in support list:\n{}".format(
            default_models, user_list, support_model_list))

    return intersection

def check_if_input_cn(prompt: str):
    if re.search("[\u4e00-\u9FFF]", prompt):
        return True
    else:
        return False

# Generator function
def image_generator(prompt, cols):
    for idx, col in enumerate(cols):
        yield generate_llm_image, (prompt, col, idx // 2)

# main entry point for serve
# python -m streamlit run image_generation.py --server.port 8088

# main entry point for debugging
# DEBUG=true python -m streamlit run image_generation.py --server.port 8088
if __name__ == "__main__":
    try:
        st.set_page_config(page_title="Da Vinci", layout="wide")
        st.title("Da Vinci")

        # Sidebar logo
        st.sidebar.image("https://d0.awsstatic.com/logos/powered-by-aws.png", width=200)

        debug = os.getenv("DEBUG", "false").lower() == "true"

        # User input
        prompt = st.text_input("What image do you want to create today?", "A cute dog")
        button = st.button('Generate Image')

        # Initialize checkpoints before user action
        get_checkpoints()

        if button:
            st.session_state.warnings = []
            st.session_state.succeed_count = 0
            # 2*2 layout image grid
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            cols = [col1, col2, col3, col4]
            # Create a generator for image generation tasks
            generator = image_generator(prompt, cols)

            # Execute each image generation task
            for func, args in generator:
                try:
                    func(*args)
                except Exception as e:
                    logger.exception(e)
                    # Each task can retry 5 times
                    # When one task fails, the next will not be executed
                    break

            # output box to summarize the image grid if succeed at least one image
            if st.session_state.succeed_count > 0:
                response = get_llm_summary(prompt)
                st.text_area("Summary", value=response, height=200)

    except Exception as e:
        logger.exception(e)
        raise e
