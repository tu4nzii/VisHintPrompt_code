import json
import pandas as pd
import os

def calculate_relative_error():
    # 定义文件路径
    json_file_path = r'D:\home work\Agent.paper\Agent\coordinates_by_image_radar_GLM-4.6V-Flash_async_updated.json'
    excel_file_path = r'd:/home work/Agent.paper/Agent/rose_average results.xlsx'
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"JSON文件不存在: {json_file_path}")
        return
    
    # 读取JSON数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储结果
    results = []
    
    # 遍历每张图表
    for chart_id, chart_data in data.items():
        if 'data' in chart_data and 'r_tick_min' in chart_data and 'r_tick_max' in chart_data:
            # 获取r_tick范围
            r_tick_range = chart_data['r_tick_max'] - chart_data['r_tick_min']
            if r_tick_range == 0:
                continue  # 避免除以零
            
            chart_results = {
                'Chart.No': chart_id.split('_')[-1],
                'origin': [],  # 存储所有点的origin值用于验证
                'with_grid_errors': [],
                'no_grid_errors': [],
                'feedback_errors': [],  # 使用feedback[1]的数据
                'amplifier_feedback_errors': [],
                'amplifier_grid_errors': [],
                # 新增基于r_tick_range的指标
                'with_grid_errors_normalized': [],
                'no_grid_errors_normalized': [],
                'feedback_errors_normalized': [],
                'amplifier_feedback_errors_normalized': [],
                'amplifier_grid_errors_normalized': []
            }
            
            # 遍历图表中的每个点
            for point_id, point_data in chart_data['data'].items():
                if 'origin' in point_data:
                    origin_value = point_data['origin']
                    if origin_value == 0:
                        continue  # 避免除以零
                    
                    chart_results['origin'].append(origin_value)
                    
                    # 计算with_grid的相对误差
                    if 'with_grid' in point_data:
                        if point_data['with_grid'] == None:
                            continue  # 避免除以零
                        grid_error = abs(origin_value - point_data['with_grid']) / origin_value
                        # if grid_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "grid_error:", grid_error)
                        #     continue  # 跳过异常值
                        chart_results['with_grid_errors'].append(grid_error)
                        #print(f"with_grid 相对误差: {grid_error:.3f}")
                        
                        # 计算基于r_tick_range的标准化误差
                        grid_error_normalized = abs(origin_value - point_data['with_grid']) / r_tick_range
                        if grid_error_normalized > 1:
                            print(chart_results['Chart.No'], point_id, "grid_error_normalized:", grid_error_normalized)
                            #continue  # 跳过异常值
                        chart_results['with_grid_errors_normalized'].append(grid_error_normalized)
                        
                    
                    # 计算no_grid的相对误差
                    if 'no_grid' in point_data:
                        if point_data['no_grid'] == None:
                            continue  # 避免除以零
                        no_grid_error = abs(origin_value - point_data['no_grid']) / origin_value
                        # if no_grid_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "no_grid_error:", no_grid_error)
                        #     continue  # 跳过异常值
                        chart_results['no_grid_errors'].append(no_grid_error)
                        
                        # 计算基于r_tick_range的标准化误差
                        no_grid_error_normalized = abs(origin_value - point_data['no_grid']) / r_tick_range
                        if no_grid_error_normalized > 1:
                            print(chart_results['Chart.No'], point_id, "no_grid_error_normalized:", no_grid_error_normalized)
                            #continue  # 跳过异常值
                        chart_results['no_grid_errors_normalized'].append(no_grid_error_normalized)
                    
                    # 计算feedback的相对误差（使用feedback[-1]）
                    if 'feedback' in point_data and isinstance(point_data['feedback'], list) and len(point_data['feedback']) > 1:
                        feedback_error = abs(origin_value - point_data['feedback'][-1]) / origin_value
                        # if feedback_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "feedback_error:", feedback_error)
                        #     continue  # 跳过异常值
                        chart_results['feedback_errors'].append(feedback_error)
                        
                        # 计算基于r_tick_range的标准化误差
                        feedback_error_normalized = abs(origin_value - point_data['feedback'][-1]) / r_tick_range
                        if feedback_error_normalized > 1:
                            print(chart_results['Chart.No'], point_id, "feedback_error_normalized:", feedback_error_normalized)
                            #continue  # 跳过异常值
                        chart_results['feedback_errors_normalized'].append(feedback_error_normalized)
                    
                    # 计算amplifier_feedback的相对误差
                    if 'amplifier_feedback_ticks' in point_data and isinstance(point_data['amplifier_feedback_ticks'], list) and len(point_data['amplifier_feedback_ticks']) > 0:
                        if point_data['amplifier_feedback_ticks'][-1]:
                            amp_feedback_error = abs(origin_value - point_data['amplifier_feedback_ticks'][-1]) / origin_value
                            # if amp_feedback_error > 5:
                            #         print(chart_results['Chart.No'], point_id, "amp_feedback_error:", amp_feedback_error)
                                    #continue  # 跳过异常值
                            amp_feedback_error_normalized = abs(origin_value - point_data['amplifier_feedback_ticks'][-1]) / r_tick_range
                            if amp_feedback_error_normalized > 1:
                                print(chart_results['Chart.No'], point_id, "amp_feedback_error_normalized:", amp_feedback_error_normalized)
                                #continue  # 跳过异常值
                        else:
                            # 确保feedback存在且不为空
                            if 'feedback' in point_data and isinstance(point_data['feedback'], list) and len(point_data['feedback']) > 0:
                                amp_feedback_error = abs(origin_value - point_data['feedback'][-1]) / origin_value
                                # if amp_feedback_error > 5:
                                #     print(chart_results['Chart.No'], point_id, "amp_feedback_error:", amp_feedback_error)
                                    #continue  # 跳过异常值
                                amp_feedback_error_normalized = abs(origin_value - point_data['feedback'][-1]) / r_tick_range
                                if amp_feedback_error_normalized > 1:
                                    print(chart_results['Chart.No'], point_id, "amp_feedback_error_normalized:", amp_feedback_error_normalized)
                                    #continue  # 跳过异常值
                            else:
                                continue  # 如果没有可用的数据，跳过这个点
                        # if amp_feedback_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "amp_feedback_error:", amp_feedback_error)
                        #     continue  # 跳过异常值
                        chart_results['amplifier_feedback_errors'].append(amp_feedback_error)
                        chart_results['amplifier_feedback_errors_normalized'].append(amp_feedback_error_normalized)
                    elif 'feedback' in point_data and isinstance(point_data['feedback'], list) and len(point_data['feedback']) > 0:
                        # 如果amplifier_feedback_ticks不可用，回退到使用feedback
                        amp_feedback_error = abs(origin_value - point_data['feedback'][-1]) / origin_value
                        # if amp_feedback_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "amp_feedback_error:", amp_feedback_error)
                        #     continue  # 跳过异常值
                        chart_results['amplifier_feedback_errors'].append(amp_feedback_error)
                            
                        # 计算基于r_tick_range的标准化误差
                        amp_feedback_error_normalized = abs(origin_value - point_data['feedback'][-1]) / r_tick_range
                        if amp_feedback_error_normalized > 5:
                            print(chart_results['Chart.No'], point_id, "amp_feedback_error_normalized:", amp_feedback_error_normalized)
                            #continue  # 跳过异常值
                        chart_results['amplifier_feedback_errors_normalized'].append(amp_feedback_error_normalized)
                    
                    # 计算amplifier_grid的相对误差
                    if 'amplifier_grid_ticks' in point_data and isinstance(point_data['amplifier_grid_ticks'], list) and len(point_data['amplifier_grid_ticks']) > 0:
                        if point_data['amplifier_grid_ticks'][-1]:
                            amp_grid_error = abs(origin_value - point_data['amplifier_grid_ticks'][-1]) / origin_value
                            amp_grid_error_normalized = abs(origin_value - point_data['amplifier_grid_ticks'][-1]) / r_tick_range
                            # if amp_grid_error > 5:
                            #     print(chart_results['Chart.No'], "amp_grid_error:", amp_grid_error)
                            #     continue  # 跳过异常值
                            if amp_grid_error_normalized > 1:
                                print(chart_results['Chart.No'], point_id, "amp_grid_error_normalized:", amp_grid_error_normalized)
                        else:
                            # 确保with_grid存在且有效
                            if 'with_grid' in point_data and point_data['with_grid'] is not None:
                                amp_grid_error = abs(origin_value - point_data['with_grid']) / origin_value
                                amp_grid_error_normalized = abs(origin_value - point_data['with_grid']) / r_tick_range
                                # if amp_grid_error > 5:
                                #     print(chart_results['Chart.No'], point_id, "amp_grid_error:", amp_grid_error)
                                #     continue  # 跳过异常值
                                if amp_grid_error_normalized > 1:
                                    print(chart_results['Chart.No'], point_id, "amp_grid_error_normalized:", amp_grid_error_normalized)
                                    #continue  # 跳过异常值
                            else:
                                continue  # 如果没有可用的数据，跳过这个点
                        chart_results['amplifier_grid_errors'].append(amp_grid_error)
                        chart_results['amplifier_grid_errors_normalized'].append(amp_grid_error_normalized)
                    elif 'with_grid' in point_data and point_data['with_grid'] is not None:

                        # 如果amplifier_grid_ticks不可用，回退到使用with_grid
                        amp_grid_error = abs(origin_value - point_data['with_grid']) / origin_value
                        # if amp_grid_error > 5:
                        #     print(chart_results['Chart.No'], point_id, "amp_grid_error:", amp_grid_error)
                        #     continue  # 跳过异常值
                        chart_results['amplifier_grid_errors'].append(amp_grid_error)
                        
                        # 计算基于r_tick_range的标准化误差
                        amp_grid_error_normalized = abs(origin_value - point_data['with_grid']) / r_tick_range
                        if amp_grid_error_normalized > 5:
                            print(chart_results['Chart.No'], point_id, "amp_grid_error_normalized:", amp_grid_error_normalized)
                            #continue  # 跳过异常值
                        chart_results['amplifier_grid_errors_normalized'].append(amp_grid_error_normalized)
            
            # 计算每种类型的平均相对误差
            if chart_results['origin']:  # 确保有点数据
                result_row = {
                    'Chart.No': chart_results['Chart.No'],
                    'baseline': round(sum(chart_results['no_grid_errors']) / len(chart_results['no_grid_errors']), 3) if chart_results['no_grid_errors'] else 0,
                    'grid': round(sum(chart_results['with_grid_errors']) / len(chart_results['with_grid_errors']), 3) if chart_results['with_grid_errors'] else 0,
                    'feedback': round(sum(chart_results['feedback_errors']) / len(chart_results['feedback_errors']), 3) if chart_results['feedback_errors'] else 0,
                    'amplifier_feedback': round(sum(chart_results['amplifier_feedback_errors']) / len(chart_results['amplifier_feedback_errors']), 3) if chart_results['amplifier_feedback_errors'] else 0,
                    'amplifier_grid': round(sum(chart_results['amplifier_grid_errors']) / len(chart_results['amplifier_grid_errors']), 3) if chart_results['amplifier_grid_errors'] else 0,
                    # 新增基于r_tick_range的标准化误差指标
                    'baseline_normalized': round(sum(chart_results['no_grid_errors_normalized']) / len(chart_results['no_grid_errors_normalized']), 3) if chart_results['no_grid_errors_normalized'] else 0,
                    'grid_normalized': round(sum(chart_results['with_grid_errors_normalized']) / len(chart_results['with_grid_errors_normalized']), 3) if chart_results['with_grid_errors_normalized'] else 0,
                    'feedback_normalized': round(sum(chart_results['feedback_errors_normalized']) / len(chart_results['feedback_errors_normalized']), 3) if chart_results['feedback_errors_normalized'] else 0,
                    'amplifier_feedback_normalized': round(sum(chart_results['amplifier_feedback_errors_normalized']) / len(chart_results['amplifier_feedback_errors_normalized']), 3) if chart_results['amplifier_feedback_errors_normalized'] else 0,
                    'amplifier_grid_normalized': round(sum(chart_results['amplifier_grid_errors_normalized']) / len(chart_results['amplifier_grid_errors_normalized']), 3) if chart_results['amplifier_grid_errors_normalized'] else 0
                }
                results.append(result_row)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按Chart.No排序
    df = df.sort_values('Chart.No')
    
    # 计算平均值行
    avg_row = df.mean(numeric_only=True)
    avg_row['Chart.No'] = 'Average'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 检查Excel文件是否存在
    if os.path.exists(excel_file_path):
        # 读取现有Excel文件
        try:
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # 写入到第一个sheet
                df.to_excel(writer, sheet_name='radar_GLM-4.6V-Flash', index=False)
        except Exception as e:
            print(f"写入现有Excel文件时出错: {e}")
            # 尝试创建新文件
            df.to_excel(excel_file_path, sheet_name='Sheet1', index=False)
    else:
        # 创建新的Excel文件
        df.to_excel(excel_file_path, sheet_name='Sheet1', index=False)
    
    print(f"计算完成！结果已保存到: {excel_file_path}")
    print(f"处理了 {len(results)} 张图表")

if __name__ == "__main__":
    calculate_relative_error()