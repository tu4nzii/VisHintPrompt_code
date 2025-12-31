import json
import os
from typing import List, Dict, Tuple
import base64
import requests
import re
import time
import cv2
import math
import numpy as np
import concurrent.futures
import threading

# 尝试导入tqdm用于进度条，若无则定义占位符
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
url = "https://api.vveai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# json 数据库导入
llm_model = "gemini-2.0-flash"

# --- 线程锁 ---
result_lock = threading.Lock()
print_lock = threading.Lock()

def hex_to_rgb(hex_color):
    """
    将十六进制颜色值转换为RGB元组
    """
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
    line_color = (100, 100, 100) # 深灰色
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

def crop_axis_label_region(image_path, center_x, center_y, angle_deg, outer_radius, angle_width=30, inner_radius=0, label_offset=30, scale_factor=2.0, r_ticks=[], arg_a=0, arg_b=0):
    # 确保文件读取安全
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None
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

    base_font_scale = 0.35
    final_font_scale = base_font_scale * (scale_factor ** 0.5) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_color = (0, 0, 0)
    font_thickness = 1 
    
    label_ext_deg = 4      
    text_rad_offset = 2 * scale_factor 

    for tick in interpolated_ticks:
        radius_scaled = int(scaled_arg_a * tick + scaled_arg_b)
        
        if radius_scaled <= 0 or radius_scaled >= scaled_outer_radius or radius_scaled <= scaled_inner_radius:
            continue
        
        final_text_radius = radius_scaled + text_rad_offset
        
        text = str(int(tick)) if tick % 1 == 0 else f"{tick:.2f}"
        
        (t_w, t_h), baseline = cv2.getTextSize(text, font, final_font_scale, font_thickness)
        
        ang_start_rad = math.radians(start_angle - label_ext_deg)
        sx = scaled_center_x + final_text_radius * math.cos(ang_start_rad)
        sy = scaled_center_y - final_text_radius * math.sin(ang_start_rad)
        
        cv2.putText(crop_img_scaled, text, (int(sx - t_w/2), int(sy + t_h/2)), 
                    font, final_font_scale, font_color, font_thickness, cv2.LINE_AA)

        ang_end_rad = math.radians(end_angle + label_ext_deg)
        ex = scaled_center_x + final_text_radius * math.cos(ang_end_rad)
        ey = scaled_center_y - final_text_radius * math.sin(ang_end_rad)
        
        cv2.putText(crop_img_scaled, text, (int(ex - t_w/2), int(ey + t_h/2)), 
                    font, final_font_scale, font_color, font_thickness, cv2.LINE_AA)

    return crop_img_scaled

def draw_angle_indicator(image, center_x, center_y, target_angle, radius, arc_color=(0, 0, 255), line_color=(0, 0, 255), 
                         arc_thickness=2, line_thickness=2, arc_angle_width=10, line_length_ratio=0.3):
    start_angle =  - target_angle - arc_angle_width // 2
    end_angle =  - target_angle + arc_angle_width // 2  
    
    cv2.ellipse(image, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, arc_color, arc_thickness, lineType=cv2.LINE_AA)
    
    angle_rad = math.radians(target_angle)
    outer_x = int(center_x + (radius+line_length_ratio*radius) * math.cos(angle_rad))
    outer_y = int(center_y - (radius+line_length_ratio*radius) * math.sin(angle_rad))
    
    inner_radius = radius * (1 - line_length_ratio)
    inner_x = int(center_x + inner_radius * math.cos(angle_rad))
    inner_y = int(center_y - inner_radius * math.sin(angle_rad))
    
    cv2.line(image, (outer_x, outer_y), (inner_x, inner_y), line_color, line_thickness, lineType=cv2.LINE_AA)
    
    return image


def find_point(image, color):
    "传入图片，给大模型，判断其中是否包含需要的颜色点，若包含，返回True，不包含，返回False"
    if image is None or image.size == 0:
        return "False"
        
    max_retries = 3
    retry_delay = 1 
    
    for attempt in range(max_retries):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            retval, buffer = cv2.imencode('.png', rgb_image)
            if not retval:
                return None
                
            base64_image = base64.b64encode(buffer).decode('utf-8')
            prompt = f"判断图片中是否包含颜色{color}返回True，不包含，返回False"
            
            payload = {
                "model": "gpt-4o",
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
            
        except Exception:
            time.sleep(retry_delay)
            continue
    
    return None

def generate_prompt(item_name: str, prompt_type: str, dataset: dict, tick=0) -> str:
    chart_type = dataset.get('chart_type', '')
    if prompt_type == "with_grid":
        if chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）

你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少
以下为提示：
    1. **在玫瑰图上找到"{item_name}"对应的扇形区域，即确定其角度范围。**,非常重要，以x轴正方向为0度，可以理解为{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，{dataset.get('theta_ticks')[1]}对应范围为{dataset.get('theta_angles')[1]}-{dataset.get('theta_angles')[2]}度，以此类推
    例子：该图的{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，处在图最右端
    2. 确定其径向位置，找到其处于哪两个网格线之间，网格线包含以下刻度{dataset.get('r_ticks', [])}，必须准确的识别其位于哪两个网格线之间
    例子：该图的"{dataset.get('theta_ticks')[0]}"的值就为{dataset['data_points'][dataset.get('theta_ticks')[0]]}
    3. 根据其扇形和相对于两个网格线的位置，插值计算其数据值。

**记住，一定要插值，利用好网格线的刻度值**
在预测之前，再次回顾以下我给你的提示
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "no_grid":
        if chart_type == 'rose':
            return f'''
您的任务是估计标记为"{item_name}"对应的值：

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "feedback":
        if chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）
你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少

**重要提示**：图表中已添加红色十字，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色十字与真实数据点的位置差距，重新优化您的预测：
1. 确定红色十字与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

**记住，一定要利用网格线进行精确插值**

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "amplifier":
        if chart_type == 'rose':
            return f'''
你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少
-请先找到{item_name}对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形，**并且找到其最远端**
-然后找到该边界处于哪两个基准线之间
-最后依据基准线的数值，插值出数值
一定要是颜色的最远端！！
警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    return ""

def call_llm_response(prompt: str, image_path: str, item_name: str, dataset: dict) -> Tuple[float | None, float | None]:
    if not os.path.exists(image_path):
        return (None, None)

    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception:
        return (None, None)

    max_retries = 15
    retry_delay = 0.5 
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
            response = requests.post(url=url, headers=headers, json=payload, timeout=20)
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
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)

    with print_lock:
        print(f"警告 已达到最大重试次数 ({max_retries}次) for {item_name}")
    return (None, None)

# --- 核心逻辑封装 ---

def process_single_item(task_info: Dict):
    """
    处理单个 Item 的完整流程 (Rose 版)
    """
    dataset = task_info['dataset']
    grid_type = task_info['grid_type']
    item_name = task_info['item_name']
    origin_value = task_info['value']
    results_by_image = task_info['results_container']
    dirs = task_info['dirs']
    
    chart_id = dataset['chart_id']
    image_path = dataset["image_paths"][grid_type].replace('\\', '/')
    
    # 1. 确保 results 结构
    with result_lock:
        if chart_id not in results_by_image:
            results_by_image[chart_id] = {
                'chart_type': dataset['chart_type'],
                'data': {}
            }
        if item_name not in results_by_image[chart_id]['data']:
            results_by_image[chart_id]['data'][item_name] = {}

    # 2. 初始预测
    color = dataset.get('series_color', {}).get(item_name, '未知颜色')
    rgb_colors = hex_to_rgb(color)
    
    prompt = generate_prompt(item_name, grid_type, dataset)
    coords = call_llm_response(prompt, image_path, item_name, dataset)

    # 结果暂存
    item_results = {}
    
    # 3. 处理 Feedback 和 Amplifier
    axis_labels = dataset.get('axis_labels')
    label_to_angle = {v: int(k) for k, v in axis_labels.items()}
    item_label = item_name

    # 检查 Item 是否在标签中
    if item_label not in label_to_angle:
        with print_lock:
            print(f"警告: 未找到{item_label}对应的角度，跳过 {chart_id}")
        return

    angle_width = int(360 / len(dataset['axis_labels']))
    target_angle = label_to_angle[item_label]

    feedback_tick = []
    amplifier_feedback_ticks = []
    amplifier_grid_ticks = []
    
    # Feedback 流程
    if grid_type == 'with_grid':
        feedback_counts = 1 # 原代码为 1
        
        if coords[0] is not None:
            feedback_tick.append(coords[0])
        else:
            feedback_tick.append(origin_value)
            
        feedback_times = 3
        
        # 只读取一次原图
        if os.path.exists(dataset["image_paths"]['with_grid'].replace('\\', '/')):
            base_img = cv2.imread(dataset["image_paths"]['with_grid'].replace('\\', '/'))
        else:
            base_img = None

        if base_img is not None:
            while feedback_counts > 0:
                feedback_image = base_img.copy()
                # 唯一文件名
                f_name = f'{chart_id}_{grid_type}_{item_name.replace(" ", "_")}_{feedback_times}.png'
                feedback_image_path = os.path.join(dirs['feedback'], f_name)
                
                center_x = dataset["pred_coords"][0]
                center_y = dataset["pred_coords"][1]
                a = dataset["argument"]["a"]
                b = dataset["argument"]["b"]
                
                curr_val = feedback_tick[-1] if feedback_tick[-1] is not None else feedback_tick[0]
                pre_r = int(a * curr_val + b)
                
                feedback_image = draw_angle_indicator(feedback_image, center_x, center_y, target_angle, pre_r, 
                                                      line_thickness=2, arc_angle_width=angle_width, line_length_ratio=0.05)
                cv2.imwrite(feedback_image_path, feedback_image)
                
                feedback_prompt = generate_prompt(item_name, 'feedback', dataset, curr_val)
                feedback_coords = call_llm_response(feedback_prompt, feedback_image_path, item_name, dataset)
                
                if feedback_coords[0] is not None:
                    feedback_tick.append(feedback_coords[0])
                elif coords[0] is not None:
                    feedback_tick.append(coords[0])
                
                # 清理
                if os.path.exists(feedback_image_path):
                    os.remove(feedback_image_path)
                    
                feedback_times += 1
                feedback_counts -= 1
        
        if feedback_tick:
            item_results['feedback'] = feedback_tick

        # Amplifier 流程 (Wit Grid Only)
        amplifier_counts = 3
        amplifier_path = dataset["image_paths"]['no_grid'].replace('\\', '/')
        
        center_x, center_y = dataset["pred_coords"]
        arg_a = dataset["argument"]["a"]
        arg_b = dataset["argument"]["b"]
        r_ticks = dataset["r_ticks"]
        max_radius = r_ticks[-1] * arg_a + arg_b
        
        current_feedback_radius = int(arg_a * feedback_tick[-1] + arg_b) if feedback_tick and feedback_tick[-1] is not None else 0
        current_grid_radius = int(arg_a * coords[0] + arg_b) if coords and coords[0] is not None else 0
        
        max_radius_pixel = (r_ticks[-1] - r_ticks[0]) * arg_a + arg_b
        amplifier_outter_interval = max_radius_pixel / len(r_ticks)
        amplifier_inner_interval = max_radius_pixel / len(r_ticks)
        
        for amp_count in range(amplifier_counts):
            # 辅助函数：计算Radius范围
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
            # 唯一文件名
            safe_name = item_name.replace(" ", "_")
            amp_fb_path = os.path.join(dirs['amp_feedback'], f'{chart_id}_{grid_type}_{safe_name}_amp{amp_count}.png')
            amp_gd_path = os.path.join(dirs['amp_grid'], f'{chart_id}_{grid_type}_{safe_name}_amp{amp_count}.png')
            
            # 定义裁剪+找点逻辑 (Rose 版本 crop_axis_label_region)
            def process_amp_image(path, out_r, in_r, save_path):
                img = crop_axis_label_region(amplifier_path, center_x, center_y, target_angle, out_r, angle_width, in_r, 30, scale_factor, r_ticks, arg_a, arg_b)
                
                curr_out = out_r
                curr_in = in_r
                loop_limit = 5
                
                # 循环寻找点
                while loop_limit > 0 and find_point(img, rgb_colors) == "False" and curr_out < max_radius:
                    curr_out += amplifier_outter_interval
                    curr_in -= amplifier_inner_interval
                    if curr_in < 0: curr_in = 0
                    img = crop_axis_label_region(amplifier_path, center_x, center_y, target_angle, curr_out, angle_width, curr_in, 30, scale_factor, r_ticks, arg_a, arg_b)
                    loop_limit -= 1
                
                if img is not None and img.size > 0:
                    cv2.imwrite(save_path, img)
                    return True
                return False

            # 执行裁剪
            fb_ok = process_amp_image(amplifier_path, if_outer, if_inner, amp_fb_path)
            gd_ok = process_amp_image(amplifier_path, ig_outer, ig_inner, amp_gd_path)
            
            if not fb_ok or not gd_ok:
                 # Fallback logic
                 if amp_count == 0: break
                 if amplifier_feedback_ticks: amplifier_feedback_ticks.append(amplifier_feedback_ticks[-1])
                 if amplifier_grid_ticks: amplifier_grid_ticks.append(amplifier_grid_ticks[-1])
                 continue

            # API Calls
            amplifier_prompt = generate_prompt(item_name, 'amplifier', dataset)
            # 这里串行调用两次API，在当前worker线程内
            amp_fb_coords = call_llm_response(amplifier_prompt, amp_fb_path, item_name, dataset)
            amp_gd_coords = call_llm_response(amplifier_prompt, amp_gd_path, item_name, dataset)
            
            # 更新结果
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

            # 清理临时文件
            # if os.path.exists(amp_fb_path): os.remove(amp_fb_path)
            # if os.path.exists(amp_gd_path): os.remove(amp_gd_path)

        if amplifier_feedback_ticks:
            item_results['amplifier_feedback_ticks'] = amplifier_feedback_ticks
        if amplifier_grid_ticks:
            item_results['amplifier_grid_ticks'] = amplifier_grid_ticks

    # 4. 更新全局结果 (加锁)
    with result_lock:
        target_entry = results_by_image[chart_id]['data'][item_name]
        
        if coords is not None:
            target_entry[grid_type] = coords[0]
        
        target_entry['origin'] = origin_value
        
        if grid_type == 'with_grid':
            if 'feedback' in item_results:
                target_entry['feedback'] = item_results['feedback']
            if 'amplifier_feedback_ticks' in item_results:
                target_entry['amplifier_feedback_ticks'] = item_results['amplifier_feedback_ticks']
            if 'amplifier_grid_ticks' in item_results:
                target_entry['amplifier_grid_ticks'] = item_results['amplifier_grid_ticks']

    # 打印日志 (可选)
    # with print_lock:
    #      print(f"Processed {item_name} ({grid_type})")

if __name__ == '__main__':
    # 1. 加载数据集
    try:
        with open(f'evaluation_datasets_with_axes_rose.json', 'r', encoding='utf-8') as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Dataset not found.")
        exit(1)
   
    # 2. 准备目录
    feedback_image_dir = './data/feedback'
    amplifier_feedback_image_dir = './data/amplifier/rose/feedback'
    amplifier_grid_image_dir = './data/amplifier/rose/grid'
    
    for d in [feedback_image_dir, amplifier_feedback_image_dir, amplifier_grid_image_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    dirs = {
        'feedback': feedback_image_dir,
        'amp_feedback': amplifier_feedback_image_dir,
        'amp_grid': amplifier_grid_image_dir
    }

    # 3. 生成任务列表
    tasks = []
    results_by_image = {} # 全局结果容器
    max_charts = len(datasets)
    print(f"Total charts: {max_charts}")
    # 限制处理的图表数量，原代码为 datasets[:1]
    # 如果需要处理更多，请修改这里
    chart_ids = ['rose_020']
    for dataset in datasets[:50]: 
        if dataset['chart_type'] != 'rose' or not dataset['data']:
            continue
        chart_id = dataset['chart_id']
        if chart_id not in chart_ids:
            continue
        
        for grid_type in ['with_grid', 'no_grid']:
            for key, value in dataset['data'].items():
                item_name = key
                task = {
                    'dataset': dataset,
                    'grid_type': grid_type,
                    'item_name': item_name,
                    'value': value,
                    'results_container': results_by_image,
                    'dirs': dirs
                }
                tasks.append(task)
    
    print(f"Total tasks: {len(tasks)}")

    # 4. 并发执行
    max_workers = 10 
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_single_item, tasks), total=len(tasks)))
        
    end_time = time.time()
    print(f"Execution finished in {end_time - start_time:.2f}s")
    
    # 5. 保存结果
    output_filename = f'coordinates_by_image_rose_{llm_model}.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_by_image, f, ensure_ascii=False, indent=4)
        
    print(f"Saved to {output_filename}")