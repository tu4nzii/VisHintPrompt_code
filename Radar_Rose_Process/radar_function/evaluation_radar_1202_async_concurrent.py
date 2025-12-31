import json
import os
from typing import List, Dict, Tuple, Any
import base64
import re
import time
import cv2
import math
import numpy as np
import asyncio
import aiohttp
import random

try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# --- 全局配置 ---
api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
# 注意：aiohttp 请求时通常只需要 base url，或者完整 url。这里保持你的完整 url
api_url = "https://api.vveai.com/v1/chat/completions" 
headers = {
    "Content-Type": "application/json",                                                                                                      
    "Authorization": f"Bearer {api_key}"
}
llm_model = "gemini-2.0-flash"   

# --- 并发控制配置 ---
# 建议设置为 20-50，过高会导致 API 报 429 错误
CONCURRENT_LIMIT = 30 

# --- 辅助函数 (数学逻辑保持不变) ---

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def extract_json_response(content: str):
    try:
        match = re.search(r'(\{[\s\S]*\})', content)
        if not match:
            return None
        json_str = match.group(1)
        return json.loads(json_str)
    except Exception as e:
        # print(f"ERROR JSON解析失败: {e}")
        return None

def validate_coordinates(coords: Tuple) -> bool:
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return False
    valid = lambda x: isinstance(x, (int, float)) or x is None
    return valid(coords[0]) and valid(coords[1])

# --- 图像处理函数 (改为内存操作) ---

def draw_sector_grid(image, center_x, center_y, arg_a, arg_b, r_ticks, start_angle_deg, end_angle_deg, extension_deg=5, outer_radius=300, inner_radius=0):
    # 保持原逻辑不变
    line_color = (100, 100, 100)
    thickness = 1
    draw_start = start_angle_deg - extension_deg
    draw_end = end_angle_deg + extension_deg
    local_ticks = [t for t in r_ticks if t != 0]

    for tick in local_ticks:
        radius = int(arg_a * tick + arg_b)
        if radius <= 0 or radius >= outer_radius or radius <= inner_radius: 
            continue
        circumference = 2 * math.pi * radius
        if circumference == 0: continue
        
        dash_len_px = 3
        gap_len_px = 4
        total_step_rad = (dash_len_px + gap_len_px) / radius
        dash_step_rad = dash_len_px / radius
        
        current_rad = math.radians(draw_start)
        end_rad = math.radians(draw_end)
        
        if end_rad < current_rad:
            end_rad += 2 * math.pi

        while current_rad < end_rad:
            seg_end_rad = current_rad + dash_step_rad
            if seg_end_rad > end_rad: 
                seg_end_rad = end_rad
            
            p1_x = int(center_x + radius * math.cos(current_rad))
            p1_y = int(center_y - radius * math.sin(current_rad))
            p2_x = int(center_x + radius * math.cos(seg_end_rad))
            p2_y = int(center_y - radius * math.sin(seg_end_rad))
            
            cv2.line(image, (p1_x, p1_y), (p2_x, p2_y), line_color, thickness, cv2.LINE_AA)
            current_rad += total_step_rad
    return image

def process_crop_axis_label_region_memory(source_image, center_x, center_y, angle_deg, outer_radius, angle_width=30, inner_radius=0, label_offset=30, scale_factor=2.0, r_ticks=[], arg_a=0, arg_b=0):
    if source_image is None:
        return None
    h, w = source_image.shape[:2]

    # --- 修复开始：防御性检查 ---
    # 强制将半径转换为非负整数
    out_r = max(0, int(outer_radius))
    in_r = max(0, int(inner_radius))
    
    # 如果外径为0，说明无法裁剪，直接返回原图或者空图
    if out_r == 0:
        return source_image
    # --- 修复结束 ---

    start_angle = angle_deg - angle_width / 2
    end_angle = angle_deg + angle_width / 2
    
    start_angle_cv = -end_angle
    end_angle_cv = -start_angle

    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 使用安全的 out_r 和 in_r
    cv2.ellipse(mask, (int(center_x), int(center_y)), (out_r, out_r), 
                angle=0, startAngle=start_angle_cv, endAngle=end_angle_cv, 
                color=255, thickness=-1, lineType=cv2.LINE_AA)

    if in_r > 0:
        cv2.ellipse(mask, (int(center_x), int(center_y)), (in_r, in_r), 
                    angle=0, startAngle=start_angle_cv - 5, endAngle=end_angle_cv + 5, 
                    color=0, thickness=-1, lineType=cv2.LINE_AA)

    # ... (后续代码保持不变，注意用到 radius 的地方都要确保安全，但前面已经处理了掩码，后面问题不大) ...
    # 为了安全，后续用到 scaled_outer_radius 的地方也要注意
    
    coords = cv2.findNonZero(mask)
    if coords is None: return source_image

    x, y, w_sector, h_sector = cv2.boundingRect(coords)
    pad = 30
    
    crop_x1 = max(0, x - pad)
    crop_y1 = max(0, y - pad)
    crop_x2 = min(w, x + w_sector + pad)
    crop_y2 = min(h, y + h_sector + pad)
    
    crop_img = source_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    crop_img[crop_mask == 0] = 255

    if scale_factor != 1.0:
        new_w = int(crop_img.shape[1] * scale_factor)
        new_h = int(crop_img.shape[0] * scale_factor)
        crop_img_scaled = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        crop_img_scaled = crop_img

    scaled_center_x = (center_x - crop_x1) * scale_factor
    scaled_center_y = (center_y - crop_y1) * scale_factor
    
    scaled_arg_a = arg_a * scale_factor
    scaled_arg_b = arg_b * scale_factor
    
    scaled_outer_radius = out_r * scale_factor # 使用安全的 out_r
    scaled_inner_radius = in_r * scale_factor  # 使用安全的 in_r

    interpolated_ticks = []
    for i in range(len(r_ticks)):
        interpolated_ticks.append(r_ticks[i])
        if i < len(r_ticks) - 1:
            mid_val = (r_ticks[i] + r_ticks[i+1]) / 2
            interpolated_ticks.append(mid_val)

    draw_sector_grid(
        crop_img_scaled,
        scaled_center_x,
        scaled_center_y,
        scaled_arg_a,
        scaled_arg_b,
        interpolated_ticks,
        start_angle,
        end_angle,
        extension_deg=8,
        outer_radius=scaled_outer_radius,
        inner_radius=scaled_inner_radius
    )
    
    # ... (字体绘制代码保持不变) ...
    base_font_scale = 0.3  
    final_font_scale = base_font_scale * (scale_factor ** 0.5) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_color = (0, 0, 0)
    font_thickness = 1 
    axis_offset_deg = 3.0  
    text_rad_offset = 0  

    for tick in interpolated_ticks:
        radius_scaled = int(scaled_arg_a * tick + scaled_arg_b)
        if radius_scaled <= 0 or radius_scaled >= scaled_outer_radius or radius_scaled <= scaled_inner_radius:
            continue
        final_text_radius = radius_scaled + text_rad_offset
        text = str(int(tick)) if tick % 1 == 0 else f"{tick:.2f}"
        (t_w, t_h), baseline = cv2.getTextSize(text, font, final_font_scale, font_thickness)
        target_angle_rad = math.radians(angle_deg + axis_offset_deg)
        tx = scaled_center_x + final_text_radius * math.cos(target_angle_rad)
        ty = scaled_center_y - final_text_radius * math.sin(target_angle_rad)
        cv2.putText(crop_img_scaled, text, (int(tx - t_w/2), int(ty + t_h/2)), 
                    font, final_font_scale, font_color, font_thickness, cv2.LINE_AA)

    return crop_img_scaled

def draw_angle_indicator(image, center_x, center_y, target_angle, radius, arc_color=(0, 0, 255), line_color=(0, 0, 255), 
                         arc_thickness=2, line_thickness=2, arc_angle_width=10, line_length_ratio=0.3):
    start_angle =  - target_angle - arc_angle_width // 2
    end_angle =  - target_angle + arc_angle_width // 2
    if radius <= 0:
        radius = 0
    radius = int(radius)
    cv2.ellipse(image, (int(center_x), int(center_y)), (radius, radius), 0, start_angle, end_angle, arc_color, arc_thickness, lineType=cv2.LINE_AA)
    angle_rad = math.radians(target_angle)
    outer_x = int(center_x + (radius+line_length_ratio*radius) * math.cos(angle_rad))
    outer_y = int(center_y - (radius+line_length_ratio*radius) * math.sin(angle_rad))
    inner_radius = int(radius * (1 - line_length_ratio))
    inner_x = int(center_x + inner_radius * math.cos(angle_rad))
    inner_y = int(center_y - inner_radius * math.sin(angle_rad))
    cv2.line(image, (outer_x, outer_y), (inner_x, inner_y), line_color, line_thickness, lineType=cv2.LINE_AA)
    return image

# --- 内存辅助工具 ---

def encode_cv2_to_base64(image):
    retval, buffer = cv2.imencode('.png', image)
    if not retval:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def read_file_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# --- Async HTTP 请求封装 ---

async def call_llm_response_async(session: aiohttp.ClientSession, prompt: str, image_b64: str, item_name: str, model_name=llm_model) -> Tuple[float | None, float | None]:
    if not image_b64:
        return (None, None)

    max_retries = 15
    base_delay = 1
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}} ,
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.5
    }

    for attempt in range(max_retries):
        try:
            async with session.post(api_url, json=payload, headers=headers, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    coords_json = extract_json_response(content)
                    
                    # --- 修复逻辑开始 ---
                    # 1. 确保 coords_json 是字典
                    if isinstance(coords_json, dict) and "datapoints" in coords_json:
                        # 2. 确保 datapoints 是列表
                        if isinstance(coords_json["datapoints"], list):
                            for item in coords_json["datapoints"]:
                                # 3. 【关键修复】确保 item 是字典，不是字符串
                                if not isinstance(item, dict):
                                    continue 
                                
                                if item_name in item:
                                    coords = item[item_name]
                                    if validate_coordinates(coords):
                                        return tuple(coords)
                    # --- 修复逻辑结束 ---
                    
                    # 如果格式不对或找不到数据，返回 None 触发重试（或结束）
                    return (None, None)
                
                elif response.status == 429:
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                elif response.status >= 500:
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(base_delay)
                    
        except Exception as e:
            # 捕获其他异常（如网络错误），打印日志并重试
            # print(f"Error for {item_name}: {e}") 
            wait_time = base_delay * (2 ** attempt)
            await asyncio.sleep(wait_time)

    return (None, None)

async def check_find_point_async(session: aiohttp.ClientSession, image_b64: str, color: str) -> str:
    """
    保留你的 find_point 逻辑，但改为异步请求，模型使用 gpt-4o
    """
    max_retries = 15
    retry_delay = 1
    
    prompt = f"判断图片中是否包含颜色为{color}的点,记住是一个圆点，若包含，返回True，不包含，返回False,并说出原因"
    
    payload = {
        "model": "gpt-4o", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}} ,
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.5
    }

    for attempt in range(max_retries):
        try:
            async with session.post(api_url, json=payload, headers=headers, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return "True" if "True" in content else "False"
                else:
                    await asyncio.sleep(retry_delay)
        except Exception:
            await asyncio.sleep(retry_delay)
            
    return "False" # Default to False if failed

# --- Prompt 生成器 (保持不变) ---

def generate_prompt(item_name: str, prompt_type: str, dataset: dict, tick=0) -> str:
    color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
    rgb_colors = hex_to_rgb(color)
    chart_type = dataset.get('chart_type', '')
    
    if prompt_type == "with_grid":
        return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} angle positions


1.请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
2.然后找到该颜色对应的点，并插值出数值,rgb颜色为{rgb_colors}
3.Locate the "{item_name}" data point on the radar chart.
4.Estimate its radial position by interpolating between concentric circles.** Remember to always interpolate and make good use of the encrypted grid **

警告 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
    elif prompt_type == "no_grid":
        return f'''
Your task is to estimate the value of the data point labeled "{item_name}":

警告 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
    elif prompt_type == "feedback":
        return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} positions

Your task is to estimate the value of the data point labeled "{item_name}":

**重要提示**：图表中已添加红色十字，在对应轴上，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色十字与真实数据点的位置差距，重新优化您的预测：
{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
1. 确定红色十字与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测
然后找到该颜色对应的点，并插值出数值,rgb颜色为{rgb_colors}
警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "amplifier":
        return f'''
您正在分析雷达图的一部分。该图在二维平面上使用从同一点出发的坐标轴展示多变量数据。

- 共有{len(dataset.get('series_color', {}))}个实体：{', '.join(dataset.get('series_color', {}).keys())}，分别对应颜色{', '.join(dataset.get('series_color', {}).values())}
现在的局部放大图为{item_name.split(',')[1].strip()}轴对应的局部放大
您的任务是估计标记为"{item_name}"对应的值，即{item_name.split(',')[0].strip()}对应实体颜色点的数值。
请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
若不存在该点，但存在该颜色的线，说明上一次预测的数值过小，请往大预测
若不存在该颜色，则说说明上一次预测值过大，超出范围，请往小预测
然后找到该{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')},rgb颜色为{rgb_colors}对应的点，并插值出数值，记住，是一个圆点！

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    return ""

# --- 核心逻辑 ---

async def async_crop_and_find(
    session: aiohttp.ClientSession, 
    executor, 
    base_image_cv, 
    dataset: dict, 
    start_outer_radius: float, 
    start_inner_radius: float,
    angle: float,
    intervals: Tuple[float, float], 
    max_radius: float,
    item_color_rgb
):
    center_x, center_y = dataset["pred_coords"]
    arg_a, arg_b = dataset["argument"]["a"], dataset["argument"]["b"]
    r_ticks = dataset["r_ticks"]
    
    # --- 【关键修复】传入前也做一次保护 ---
    curr_out = max(0, float(start_outer_radius))
    curr_in = max(0, float(start_inner_radius))
    
    outer_interval, inner_interval = intervals
    angle_width = 30
    loop_limit = 5
    loop = asyncio.get_running_loop()
    final_img_b64 = None
    
    for _ in range(loop_limit):
        def crop_task():
            # 这里的 process_crop_axis_label_region_memory 已经是修复版了
            img = process_crop_axis_label_region_memory(
                base_image_cv, center_x, center_y, angle, curr_out, angle_width, curr_in, 
                30, 3.0, r_ticks, arg_a, arg_b
            )
            return img

        try:
            cropped_img = await loop.run_in_executor(executor, crop_task)
            img_b64 = await loop.run_in_executor(executor, encode_cv2_to_base64, cropped_img)
            
            if not img_b64:
                break # 图片编码失败，可能原图是空的
                
            final_img_b64 = img_b64
            
            check_result = await check_find_point_async(session, img_b64, item_color_rgb)
            if check_result == "True":
                return img_b64
                
        except Exception as e:
            # 如果这里报错（比如CV崩溃），打印日志但不中断程序，尝试下一次循环或返回 None
            print(f"Warning: Crop/Check failed: {e}")
            break

        if curr_out >= max_radius:
            break
            
        curr_out += outer_interval
        curr_in -= inner_interval
        if curr_in < 0: curr_in = 0
    
    return final_img_b64

async def process_single_task(sem: asyncio.Semaphore, session: aiohttp.ClientSession, executor, task_info: Dict):
    dataset = task_info['dataset']
    grid_type = task_info['grid_type']
    item_name = task_info['item_name']
    origin_value = task_info['value']
    
    chart_id = dataset['chart_id']
    image_path = dataset["image_paths"][grid_type].replace('\\', '/')

    # 结果容器
    result_data = {
        'origin': origin_value,
        'initial': None
    }
    
    # 异步读取主图
    if not os.path.exists(image_path):
        return (chart_id, item_name, grid_type, result_data)

    loop = asyncio.get_running_loop()
    # 读图为 B64 用于发送，同时读图为 CV2 对象用于后续处理
    base_img_b64_fut = loop.run_in_executor(executor, read_file_to_base64, image_path)
    
    # 只有 with_grid 需要 CV2 对象进行后续的画图操作
    base_img_cv_fut = None
    if grid_type == 'with_grid':
        base_img_cv_fut = loop.run_in_executor(executor, cv2.imread, image_path)
        
    base_img_b64 = await base_img_b64_fut
    
    # --- Step 1: Initial Prediction ---
    initial_prompt = generate_prompt(item_name, grid_type, dataset)
    
    async with sem:
        coords = await call_llm_response_async(session, initial_prompt, base_img_b64, item_name)
    
    result_data['initial'] = coords[0] if coords else None
    
    # 如果是 no_grid，到此结束
    if grid_type != 'with_grid':
        return (chart_id, item_name, grid_type, result_data)

    # --- Step 2: Feedback & Amplifier (Only for with_grid) ---
    
    # 获取 CV2 图片对象
    base_img_cv = await base_img_cv_fut
    if base_img_cv is None:
        return (chart_id, item_name, grid_type, result_data)

    # Feedback 配置
    feedback_tick = []
    if coords[0] is not None:
        feedback_tick.append(coords[0])
    else:
        feedback_tick.append(origin_value)

    axis_labels = dataset.get('axis_labels')
    label_to_angle = {v: int(k) for k, v in axis_labels.items()}
    item_label = item_name.split(',')[1].strip()
    target_angle = label_to_angle.get(item_label, 0)
    
    center_x, center_y = dataset["pred_coords"]
    a, b = dataset["argument"]["a"], dataset["argument"]["b"]
    
    # Feedback Loop (串行依赖)
    feedback_counts = 3
    for _ in range(feedback_counts):
        curr_val = feedback_tick[-1] if feedback_tick[-1] is not None else feedback_tick[0]
        pre_r = int(a * curr_val + b)
        
        # 在 Executor 中画图
        def draw_task():
            temp = base_img_cv.copy()
            draw_angle_indicator(temp, center_x, center_y, target_angle, pre_r, 
                               arc_thickness=1, line_thickness=1, arc_angle_width=20, line_length_ratio=0.1)
            return encode_cv2_to_base64(temp)
            
        fb_img_b64 = await loop.run_in_executor(executor, draw_task)
        fb_prompt = generate_prompt(item_name, 'feedback', dataset, curr_val)
        
        async with sem:
            fb_coords = await call_llm_response_async(session, fb_prompt, fb_img_b64, item_name)
            
        if fb_coords[0] is not None:
            feedback_tick.append(fb_coords[0])
        else:
            feedback_tick.append(feedback_tick[-1])
            
    result_data['feedback'] = feedback_tick

    # Amplifier Logic
    # 准备 Amplifier 需要的资源
    no_grid_path = dataset["image_paths"]['no_grid'].replace('\\', '/')
    # 读取 no_grid 图片用于裁剪 (只读一次)
    no_grid_img_cv = await loop.run_in_executor(executor, cv2.imread, no_grid_path)
    
    if no_grid_img_cv is not None:
        color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
        # rgb_colors 用于 find_point
        
        r_ticks = dataset["r_ticks"]
        max_radius = r_ticks[-1]*a + b
        max_radius_pixel = (r_ticks[-1] - r_ticks[0])*a + b
        
        amplifier_outter_interval = max_radius_pixel / len(r_ticks)
        amplifier_inner_interval = max_radius_pixel / len(r_ticks)
        intervals = (amplifier_outter_interval, amplifier_inner_interval)
        
        amplifier_feedback_ticks = []
        amplifier_grid_ticks = []
        
        current_feedback_radius = int(a * feedback_tick[-1] + b) if feedback_tick[-1] is not None else 0
        current_grid_radius = int(a * coords[0] + b) if coords and coords[0] is not None else 0
        
        amplifier_counts = 3
        
        for _ in range(amplifier_counts):
            # 计算内外半径
            def get_radii(curr_rad):
                if curr_rad != 0:
                    inner = curr_rad - amplifier_outter_interval
                    outer = curr_rad + amplifier_inner_interval
                    if outer > max_radius: outer = max_radius + 50
                    if inner < 0: inner = 0
                    return inner, outer
                return 0, curr_rad
                
            fb_in, fb_out = get_radii(current_feedback_radius)
            gd_in, gd_out = get_radii(current_grid_radius)
            
            # 并发执行带 find_point 逻辑的裁剪
            # 这里调用 async_crop_and_find，它内部会进行多次 LLM 检查，直到找到点或循环结束
            task_fb = async_crop_and_find(
                session, executor, no_grid_img_cv, dataset, 
                fb_out, fb_in, target_angle, intervals, max_radius, color
            )
            task_gd = async_crop_and_find(
                session, executor, no_grid_img_cv, dataset, 
                gd_out, gd_in, target_angle, intervals, max_radius, color
            )
            
            # 等待两个裁剪任务完成 (大部分时间花在网络等待上)
            async with sem:
                 amp_fb_b64, amp_gd_b64 = await asyncio.gather(task_fb, task_gd)
            
            # 拿到符合条件的图片后，进行 Amplifier 预测
            amp_prompt = generate_prompt(item_name, 'amplifier', dataset)
            
            pred_tasks = []
            if amp_fb_b64:
                pred_tasks.append(call_llm_response_async(session, amp_prompt, amp_fb_b64, item_name))
            else:
                pred_tasks.append(asyncio.sleep(0, result=(None, None)))
                
            if amp_gd_b64:
                pred_tasks.append(call_llm_response_async(session, amp_prompt, amp_gd_b64, item_name))
            else:
                pred_tasks.append(asyncio.sleep(0, result=(None, None)))
            
            async with sem:
                results = await asyncio.gather(*pred_tasks)
            
            res_fb, res_gd = results
            
            # 更新逻辑
            if res_fb and res_fb[0] is not None:
                amplifier_feedback_ticks.append(res_fb[0])
                current_feedback_radius = int(a * res_fb[0] + b)
            elif amplifier_feedback_ticks:
                amplifier_feedback_ticks.append(amplifier_feedback_ticks[-1])
                
            if res_gd and res_gd[0] is not None:
                amplifier_grid_ticks.append(res_gd[0])
                current_grid_radius = int(a * res_gd[0] + b)
            elif amplifier_grid_ticks:
                amplifier_grid_ticks.append(amplifier_grid_ticks[-1])
        
        result_data['amplifier_feedback_ticks'] = amplifier_feedback_ticks
        result_data['amplifier_grid_ticks'] = amplifier_grid_ticks
        
    return (chart_id, item_name, grid_type, result_data)


# --- 主程序入口 ---

async def main():
    # 1. 加载数据
    try:
        with open(f'evaluation_datasets_with_axes_radar.json', 'r', encoding='utf-8') as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    # 2. 生成任务列表
    tasks_info = []
    results_by_image = {}
    process_limit = 1
    max_charts = len(datasets) 
    print(f"Loading {max_charts} charts...")
    #chart_ids = ['radar_006','radar_007','radar_008','radar_019','radar_020','radar_022','radar_023','radar_048']
    # print(f"Processing charts: {chart_ids}")
    for dataset in datasets[:process_limit]:
        chart_id = dataset['chart_id']
        #if chart_id not in chart_ids: continue
        if dataset['chart_type'] != 'radar' or not dataset['data']: continue
        if dataset['num_entities'] > 5: continue
        #print(f"Processing chart: {chart_id}")
        # 初始化结果字典结构
        if chart_id not in results_by_image:
            results_by_image[chart_id] = {
                'chart_type': dataset['chart_type'],
                'data': {}
            }
            
        for grid_type in ['with_grid', 'no_grid']:
            for top_key, nested_dict in dataset['data_points'].items():
                for sub_key, value in nested_dict.items():
                    item_name = f"{top_key},{sub_key}"
                    
                    if item_name not in results_by_image[chart_id]['data']:
                        results_by_image[chart_id]['data'][item_name] = {'origin': value}
                        
                    task = {
                        'dataset': dataset,
                        'grid_type': grid_type,
                        'item_name': item_name,
                        'value': value
                    }
                    tasks_info.append(task)

    print(f"Total tasks: {len(tasks_info)}")
    
    # 3. 异步执行
    start_time = time.time()
    # 随机打乱任务顺序, 避免扎堆请求
    random.shuffle(tasks_info) 
    # 限制并发请求数
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    # 创建线程池，用于 OpenCV 操作，核心数可根据机器配置调整
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
    # 开启 HTTP Session
    timeout = aiohttp.ClientTimeout(total=None) # 单个请求通过 headers 和参数控制超时，Session 不设总超时
    connector = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300) 
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建 Task 列表
        futures = [process_single_task(sem, session, executor, t) for t in tasks_info]
        
        processed_results = []
        # 使用 as_completed 结合 tqdm 显示进度
        for f in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Processing"):
            res = await f
            if res:
                processed_results.append(res)
    
    # 4. 汇总结果
    print("Aggregating results...")
    for chart_id, item_name, grid_type, data in processed_results:
        target = results_by_image[chart_id]['data'][item_name]
        
        # 写入数据
        if data['initial'] is not None:
            target[grid_type] = data['initial']
            
        if grid_type == 'with_grid':
            if 'feedback' in data: 
                target['feedback'] = data['feedback']
            if 'amplifier_feedback_ticks' in data: 
                target['amplifier_feedback_ticks'] = data['amplifier_feedback_ticks']
            if 'amplifier_grid_ticks' in data: 
                target['amplifier_grid_ticks'] = data['amplifier_grid_ticks']

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    # 5. 保存文件
    output_filename = f'coordinates_by_image_radar_{llm_model}_async.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_by_image, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_filename}")

if __name__ == '__main__':
    # Windows 需要设置策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())