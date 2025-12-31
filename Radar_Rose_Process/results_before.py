import json

def calculate_tick_interval(r_ticks):
    """计算刻度间隔"""
    if len(r_ticks) < 2:
        return 0
    # 假设刻度是等间距的，计算前两个刻度的差值
    return r_ticks[1] - r_ticks[0]

def calculate_r_tick_min(r_ticks):
    """根据规则计算r_tick_min"""
    if len(r_ticks) % 2 == 1:  # 奇数长度
        return float(r_ticks[0])
    else:  # 偶数长度
        interval = calculate_tick_interval(r_ticks)
        return float(r_ticks[0]) - interval

def calculate_r_tick_max(r_ticks):
    """计算r_tick_max"""
    return float(r_ticks[-1])

def main():
    # 读取evaluation_datasets_with_axes_rose.json文件
    with open('evaluation_datasets_with_axes_rose.json', 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 读取coordinates_by_image_rose_internvl3-78b_async.json文件
    with open('coordinates_by_image_rose_GLM-4.6V-Flash_async.json', 'r', encoding='utf-8') as f:
        coord_data = json.load(f)
    
    # 创建一个字典来映射chart_id到r_ticks
    chart_r_ticks = {}
    for item in eval_data:
        chart_id = item['chart_id']
        r_ticks = item['r_ticks']
        chart_r_ticks[chart_id] = r_ticks
    
    # 为coordinates_by_image_radar_gemini-2.0-flash_async.json中的每个图表添加字段
    processed_count = 0
    for chart_id, r_ticks in chart_r_ticks.items():
        if chart_id in coord_data:
            # 计算r_tick_min和r_tick_max
            r_tick_min = calculate_r_tick_min(r_ticks)
            r_tick_max = calculate_r_tick_max(r_ticks)
            
            # 添加字段
            coord_data[chart_id]['r_tick_min'] = r_tick_min
            coord_data[chart_id]['r_tick_max'] = r_tick_max
            
            print(f"为图表 {chart_id} 添加了 r_tick_min={r_tick_min}, r_tick_max={r_tick_max}")
            if r_tick_min != 0:
                print(f"警告: 图表 {chart_id} 的 r_tick_min 不为0")
            processed_count += 1
        else:
            print(f"警告: 图表 {chart_id} 在坐标文件中未找到")
    
    # 保存修改后的数据
    output_filename = 'coordinates_by_image_rose_GLM-4.6V-Flash_async_updated.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(coord_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成，共处理了 {processed_count} 个图表，结果已保存到 {output_filename}")

if __name__ == "__main__":
    main()