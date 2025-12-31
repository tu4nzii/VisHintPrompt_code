import json
import os
from typing import List, Dict, Tuple, Any
import base64
import requests
import re
import time
import cv2
import math
import numpy as np
import concurrent.futures
import threading

# 尝试导入tqdm用于显示进度条，如果没有安装则定义一个简单的占位符
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# --- 全局配置 ---
api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
url = "https://api.vveai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
llm_model = "gemini-2.0-flash"

# --- 线程锁 ---
# 用于保护 results_by_image 的写入，防止多线程冲突
result_lock = threading.Lock()
# 用于保护文件写入（如果在 process 内部写文件）或打印
print_lock = threading.Lock()

# --- 辅助函数 (保持不变) ---

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
        with print_lock:
            print(f"ERROR JSON解析失败: {e}")
        return None

def validate_coordinates(coords: Tuple) -> bool:
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return False
    valid = lambda x: isinstance(x, (int, float)) or x is None
    return valid(coords[0]) and valid(coords[1])

def is_angle_between(target, start, end):
    target = target % 360
    start = start % 360
    end = end % 360
    if start <= end:
        return start <= target <= end
    else:
        return target >= start or target <= end

def draw_sector_grid(image, center_x, center_y, arg_a, arg_b, r_ticks, start_angle_deg, end_angle_deg, extension_deg=5, outer_radius=300, inner_radius=0):
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

def crop_axis_label_region_one_tick(image_path, center_x, center_y, angle_deg, outer_radius, angle_width=30, inner_radius=0, label_offset=30, scale_factor=2.0, r_ticks=[], arg_a=0, arg_b=0):
    # 注意：cv2.imread 不是完全线程安全的，如果在读取同一文件时。但在大多操作系统上只读是安全的。
    # 为了保险，确保路径存在
    if not os.path.exists(image_path):
        raise ValueError(f"无法读取图像: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像(None): {image_path}")
    h, w = image.shape[:2]

    start_angle = angle_deg - angle_width / 2
    end_angle = angle_deg + angle_width / 2
    start_angle_cv = -end_angle
    end_angle_cv = -start_angle
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(center_x), int(center_y)), (int(outer_radius), int(outer_radius)), 
                angle=0, startAngle=start_angle_cv, endAngle=end_angle_cv, 
                color=255, thickness=-1, lineType=cv2.LINE_AA)

    if inner_radius > 0:
        cv2.ellipse(mask, (int(center_x), int(center_y)), (int(inner_radius), int(inner_radius)), 
                    angle=0, startAngle=start_angle_cv - 5, endAngle=end_angle_cv + 5, 
                    color=0, thickness=-1, lineType=cv2.LINE_AA)

    coords = cv2.findNonZero(mask)
    if coords is None: return image

    x, y, w_sector, h_sector = cv2.boundingRect(coords)
    pad = 30
    crop_x1 = max(0, x - pad)
    crop_y1 = max(0, y - pad)
    crop_x2 = min(w, x + w_sector + pad)
    crop_y2 = min(h, y + h_sector + pad)
    
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
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
    scaled_outer_radius = outer_radius * scale_factor
    scaled_inner_radius = inner_radius * scale_factor

    interpolated_ticks = []
    for i in range(len(r_ticks)):
        interpolated_ticks.append(r_ticks[i])
        if i < len(r_ticks) - 1:
            mid_val = (r_ticks[i] + r_ticks[i+1]) / 2
            interpolated_ticks.append(mid_val)

    draw_sector_grid(
        crop_img_scaled, scaled_center_x, scaled_center_y, scaled_arg_a, scaled_arg_b,
        interpolated_ticks, start_angle, end_angle, extension_deg=8, 
        outer_radius=scaled_outer_radius, inner_radius=scaled_inner_radius
    )

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

def find_point(image, color):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            retval, buffer = cv2.imencode('.png', rgb_image)
            if not retval:
                return None
                
            base64_image = base64.b64encode(buffer).decode('utf-8')
            prompt = f"判断图片中是否包含颜色为{color}的点,记住是一个圆点，若包含，返回True，不包含，返回False,并说出原因"
            
            payload = {
                "model": "gpt-4o", # 注意这里使用的是 gpt-4o，根据原代码保持不变
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} ,
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "temperature": 0.5
            }
            
            # 使用 Session 也许能稍微优化连接，但 requests 直接调用也行
            response = requests.post(url=url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                time.sleep(retry_delay)
                continue
            
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                time.sleep(retry_delay)
                continue
                
            content = result["choices"][0]["message"]["content"]
            data = "True" if "True" in content else "False"
            return data
            
        except Exception as e:
            # 捕获所有 request 异常
            # print(f"Find point attempt {attempt} failed: {e}")
            time.sleep(retry_delay)
            continue
    
    return None

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

def generate_prompt(item_name: str, prompt_type: str, dataset: dict, tick=0) -> str:
    # 保持原逻辑
    color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
    rgb_colors = hex_to_rgb(color)
    chart_type = dataset.get('chart_type', '')
    
    if prompt_type == "with_grid":
        if chart_type == 'radar':
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
        if chart_type == 'radar':
            return f'''
            
Your task is to estimate the value of the data point labeled "{item_name}":

警告 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
        elif chart_type == 'rose':
            # 代码中未处理 rose 逻辑，但保持原样
            return f'''
您的任务是估计标记为"{item_name}"对应的值：

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "feedback":
        if chart_type == 'radar':
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
        if chart_type == 'radar':
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

def call_llm_response(prompt: str, image_path: str, item_name: str, dataset: dict) -> Tuple[float | None, float | None]:
    # 增加文件存在检查
    if not os.path.exists(image_path):
        with print_lock:
             print(f"File not found: {image_path}")
        return (None, None)

    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        with print_lock:
            print(f"Read Image Error {image_path}: {e}")
        return (None, None)

    max_retries = 20
    retry_delay = 1
    retry_count = 0

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} ,
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.5
    }

    while retry_count < max_retries:
        try:
            response = requests.post(url=url, headers=headers, json=payload, timeout=60) # 稍微增加超时
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            coords_json = extract_json_response(content)
            
            if coords_json and "datapoints" in coords_json:
                for item in coords_json["datapoints"]:
                    if item_name in item:
                        coords = item[item_name]
                        if validate_coordinates(coords):
                            return tuple(coords)
            
            return (None, None)
            
        except requests.exceptions.RequestException:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
        except Exception:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)

    with print_lock:
        print(f"警告 已达到最大重试次数 ({max_retries}次) for {item_name}")
    return (None, None)

# --- 核心逻辑封装 ---

def process_single_item(task_info: Dict):
    """
    处理单个 Item 的完整流程（Initial -> Feedback -> Amplifier）
    task_info 包含: dataset, grid_type, item_name, value (origin), results_container (shared dict), directories
    """
    dataset = task_info['dataset']
    grid_type = task_info['grid_type']
    item_name = task_info['item_name']
    origin_value = task_info['value']
    results_by_image = task_info['results_container']
    
    dirs = task_info['dirs']
    feedback_image_dir = dirs['feedback']
    amplifier_feedback_image_dir = dirs['amp_feedback']
    amplifier_grid_image_dir = dirs['amp_grid']
    
    chart_id = dataset['chart_id']
    image_path = dataset["image_paths"][grid_type].replace('\\', '/')

    # 1. 确保 results 结构存在 (使用锁)
    with result_lock:
        if chart_id not in results_by_image:
            results_by_image[chart_id] = {
                'chart_type': dataset['chart_type'],
                'data': {}
            }
        if item_name not in results_by_image[chart_id]['data']:
            results_by_image[chart_id]['data'][item_name] = {
                'origin': origin_value
            }

    # 2. 初始 Prompt 和 调用
    prompt = generate_prompt(item_name, grid_type, dataset)
    coords = call_llm_response(prompt, image_path, item_name, dataset)

    # 结果容器 (线程局部)
    item_results = {}
    
    # 3. 处理 Feedback 和 Amplifier
    if grid_type == 'with_grid':
        color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
        rgb_colors = hex_to_rgb(color)
        
        # --- Feedback Loop ---
        feedback_counts = 3
        feedback_tick = []
        if coords[0] is not None:
            feedback_tick.append(coords[0])
        else:
            feedback_tick.append(origin_value) # Fallback

        feedback_times = 1
        axis_labels = dataset.get('axis_labels')
        label_to_angle = {v: int(k) for k, v in axis_labels.items()}
        item_label = item_name.split(',')[1].strip()
        target_angle = label_to_angle.get(item_label, 0)
        
        # 读取原始图片 (线程安全注意：imread 读取文件是安全的)
        # 优化：不在这里重复读，而是只读一次
        base_image = cv2.imread(image_path)
        if base_image is None:
             with print_lock:
                 print(f"Error reading image for feedback: {image_path}")
             return

        while feedback_counts > 0:
            # 复制一份用于画图
            feedback_image = base_image.copy()
            # 文件名加入 item_name 和 feedback_times 确保唯一性
            f_img_name = f'{chart_id}_{grid_type}_{item_name.replace(",","_")}_{feedback_times}.png'
            feedback_image_path = os.path.join(feedback_image_dir, f_img_name)
            
            center_x = dataset["pred_coords"][0]
            center_y = dataset["pred_coords"][1]
            a = dataset["argument"]["a"]
            b = dataset["argument"]["b"]
            
            curr_val = feedback_tick[-1] if feedback_tick[-1] is not None else feedback_tick[0]
            pre_r = int(a * curr_val + b)
            
            feedback_image = draw_angle_indicator(feedback_image, center_x, center_y, target_angle, pre_r,
                                                  arc_thickness=1, line_thickness=1, arc_angle_width=20, line_length_ratio=0.1)
            cv2.imwrite(feedback_image_path, feedback_image)
            
            feedback_prompt = generate_prompt(item_name, 'feedback', dataset, curr_val)
            feedback_coords = call_llm_response(feedback_prompt, feedback_image_path, item_name, dataset)
            
            if feedback_coords[0] is not None:
                feedback_tick.append(feedback_coords[0])
            else:
                feedback_tick.append(coords[0]) # Fallback to initial
            
            # 删除临时文件
            # if os.path.exists(feedback_image_path):
            #     os.remove(feedback_image_path)
            
            feedback_times += 1
            feedback_counts -= 1
        
        if feedback_tick:
            item_results['feedback'] = feedback_tick

        # --- Amplifier Loop ---
        amplifier_counts = 3
        amplifier_feedback_ticks = []
        amplifier_grid_ticks = []

        no_grid_path = dataset["image_paths"]['no_grid'].replace('\\', '/')
        center_x, center_y = dataset["pred_coords"]
        arg_a = dataset["argument"]["a"]
        arg_b = dataset["argument"]["b"]
        r_ticks = dataset["r_ticks"]
        max_radius = r_ticks[-1]*arg_a + arg_b
        
        current_feedback_radius = int(arg_a * feedback_tick[-1] + arg_b) if feedback_tick and feedback_tick[-1] is not None else 0
        current_grid_radius = int(arg_a * coords[0] + arg_b) if coords and coords[0] is not None else 0
        
        max_radius_pixel = (r_ticks[-1] - r_ticks[0])*arg_a + arg_b
        amplifier_outter_interval = max_radius_pixel / len(r_ticks)
        amplifier_inner_interval = max_radius_pixel / len(r_ticks)
        angle_width = 30 # from main

        for amp_count in range(amplifier_counts):
            # 局部逻辑：计算半径
            def get_radii(curr_rad):
                if curr_rad != 0:
                    inner = curr_rad - amplifier_outter_interval
                    outer = curr_rad + amplifier_inner_interval
                    if outer > max_radius: outer = max_radius + 50
                    if inner < 0: inner = 0
                    return inner, outer
                return 0, curr_rad
            
            if_inner, if_outer = get_radii(current_feedback_radius)
            ig_inner, ig_outer = get_radii(current_grid_radius)
            
            scale_factor = 3
            # 文件名确保唯一
            safe_item_name = item_name.replace(",", "_")
            amp_fb_path = os.path.join(amplifier_feedback_image_dir, f'{chart_id}_{grid_type}_{safe_item_name}_amp{amp_count}.png')
            amp_gd_path = os.path.join(amplifier_grid_image_dir, f'{chart_id}_{grid_type}_{safe_item_name}_amp{amp_count}.png')
            
            # 裁剪 + 颜色检测循环 (这部分比较耗时，但必须在线程内做)
            # 为了简化代码，这里直接调用 crop_axis_label_region_one_tick 并包含 find_point 逻辑
            # 注意：多线程读同一文件(no_grid_path)没问题
            
            def safe_crop_and_find(path, out_r, in_r, save_path):
                img = crop_axis_label_region_one_tick(no_grid_path, center_x, center_y, target_angle, out_r, angle_width, in_r, 30, scale_factor, r_ticks, arg_a, arg_b)
                # 循环寻找点
                curr_out = out_r
                curr_in = in_r
                # 限制查找循环次数防止死循环
                loop_limit = 5 
                while loop_limit > 0 and find_point(img, rgb_colors) == "False" and curr_out < max_radius:
                    curr_out += amplifier_outter_interval
                    curr_in -= amplifier_inner_interval
                    if curr_in < 0: curr_in = 0
                    img = crop_axis_label_region_one_tick(no_grid_path, center_x, center_y, target_angle, curr_out, angle_width, curr_in, 30, scale_factor, r_ticks, arg_a, arg_b)
                    loop_limit -= 1
                
                if img.size > 0:
                    cv2.imwrite(save_path, img)
                    return True
                return False

            # 并行生成两张图？既然我们已经在 worker 线程里，顺序执行即可，不必再开子线程
            fb_img_ok = safe_crop_and_find(no_grid_path, if_outer, if_inner, amp_fb_path)
            gd_img_ok = safe_crop_and_find(no_grid_path, ig_outer, ig_inner, amp_gd_path)

            if not fb_img_ok or not gd_img_ok:
                 # 处理 fallback 逻辑
                 if amplifier_feedback_ticks: amplifier_feedback_ticks.append(amplifier_feedback_ticks[-1])
                 if amplifier_grid_ticks: amplifier_grid_ticks.append(amplifier_grid_ticks[-1])
                 if amp_count == 0: break 
                 continue

            amplifier_prompt = generate_prompt(item_name, 'amplifier', dataset)
            
            # 这里是两次网络调用，可以在本线程内同步做
            amp_fb_coords = call_llm_response(amplifier_prompt, amp_fb_path, item_name, dataset)
            amp_gd_coords = call_llm_response(amplifier_prompt, amp_gd_path, item_name, dataset)

            # 记录结果并更新半径
            if amp_fb_coords and amp_fb_coords[0] is not None:
                amplifier_feedback_ticks.append(amp_fb_coords[0])
                current_feedback_radius = int(arg_a * amp_fb_coords[0] + arg_b)
            elif amplifier_feedback_ticks:
                amplifier_feedback_ticks.append(amplifier_feedback_ticks[-1])

            if amp_gd_coords and amp_gd_coords[0] is not None:
                amplifier_grid_ticks.append(amp_gd_coords[0])
                current_grid_radius = int(arg_a * amp_gd_coords[0] + arg_b)
            elif amplifier_grid_ticks:
                amplifier_grid_ticks.append(amplifier_grid_ticks[-1])

            # 清理
            # if os.path.exists(amp_fb_path): os.remove(amp_fb_path)
            # if os.path.exists(amp_gd_path): os.remove(amp_gd_path)

        if amplifier_feedback_ticks:
            item_results['amplifier_feedback_ticks'] = amplifier_feedback_ticks
        if amplifier_grid_ticks:
            item_results['amplifier_grid_ticks'] = amplifier_grid_ticks

    # 4. 更新全局结果
    with result_lock:
        target_entry = results_by_image[chart_id]['data'][item_name]
        
        # 写入初始 coords
        if coords is not None:
            target_entry[grid_type] = coords[0]
        
        # 写入 origin (虽然初始化写过，保险起见)
        target_entry['origin'] = origin_value
        
        # 写入 feedback 和 amplifier 结果
        if grid_type == 'with_grid':
            if 'feedback' in item_results:
                target_entry['feedback'] = item_results['feedback']
            if 'amplifier_feedback_ticks' in item_results:
                target_entry['amplifier_feedback_ticks'] = item_results['amplifier_feedback_ticks']
            if 'amplifier_grid_ticks' in item_results:
                target_entry['amplifier_grid_ticks'] = item_results['amplifier_grid_ticks']
    
    # 打印日志 (可选)
    # with print_lock:
    #     if grid_type == 'with_grid':
    #          print(f"Done {chart_id} {item_name}: Origin={origin_value}, Coords={coords[0]}")
    #     else:
    #          print(f"Done {chart_id} {item_name} NoGrid: {coords[0]}")


# --- 主程序 ---

if __name__ == '__main__':
    # 1. 加载数据
    try:
        with open(f'evaluation_datasets_with_axes_radar.json', 'r', encoding='utf-8') as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        exit(1)

    # 2. 创建目录
    feedback_image_dir = './data/feedback'
    amplifier_feedback_image_dir = './data/amplifier/radar/feedback'
    amplifier_grid_image_dir = './data/amplifier/radar/grid'
    
    for d in [feedback_image_dir, amplifier_feedback_image_dir, amplifier_grid_image_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    dirs = {
        'feedback': feedback_image_dir,
        'amp_feedback': amplifier_feedback_image_dir,
        'amp_grid': amplifier_grid_image_dir
    }

    # 3. 准备任务列表
    # 我们将每一个 (Dataset, GridType, Item) 视为一个独立的任务
    tasks = []
    results_by_image = {} # 共享结果容器

    max_charts = len(datasets) 
    print(f"Total charts: {max_charts}")
    
    print("Generating tasks...")
    # 保持原逻辑的过滤条件
    chart_ids = ['radar_009', 'radar_016']
    for dataset in datasets[:max_charts]:
        chart_id = dataset['chart_id']
        if chart_id not in chart_ids:
            continue
        if dataset['chart_type'] != 'radar' or not dataset['data']:
            continue
        if dataset['num_entities'] > 5:
            print(f"跳过{chart_id}，实体数大于5")
            continue
        
        # 遍历 Grid Type
        for grid_type in ['with_grid', 'no_grid']:
            # 遍历 Data Items
            for top_key, nested_dict in dataset['data'].items():
                for sub_key, value in nested_dict.items():
                    item_name = f"{top_key},{sub_key}"
                    
                    task = {
                        'dataset': dataset,
                        'grid_type': grid_type,
                        'item_name': item_name,
                        'value': value,
                        'results_container': results_by_image,
                        'dirs': dirs
                    }
                    tasks.append(task)

    print(f"Total tasks to process: {len(tasks)}")

    # 4. 并发执行
    # max_workers 建议根据你的网络情况和API限制调整。
    # API 限制如果不严，可以设为 10-20。如果容易 429，设小一点。
    max_workers = 5 
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度条
        list(tqdm(executor.map(process_single_item, tasks), total=len(tasks), unit="item"))

    end_time = time.time()
    print(f"All tasks completed in {end_time - start_time:.2f} seconds.")

    # 5. 保存结果
    output_filename = f'coordinates_by_image_radar_{llm_model}.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_by_image, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_filename}")