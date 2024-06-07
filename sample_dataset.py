import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True 

"""
Time,PR_Val,Temp_Val,Sys_Val,Resp_Val,SPO2
1038,108,-,-,38,97
1039,108,-,-,36,97
1040,108,-,-,34,97
1041,107,-,-,32,97
1042,107,-,-,30,97
"""

# Step 1: Data Preprocessing
def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    for i in raw_data.columns:
        raw_data[i] = raw_data[i].replace('-', 'N')
    return raw_data

# Step 2: Sliding Window Transformation
def create_sliding_windows(data: pd.DataFrame, window_length: int):
    sliding_windows = []   
    #for i in range(0, window_length + 1):
    sliding_windows.append(data[0:window_length])
    return sliding_windows

# Step 3: Patterned Modified Early Warning Score (PMEWS)
def calculate_mews(data: pd.DataFrame):
    def assign_mews_score_to_pulse_rate(pulse_rate):
        if 51 <= pulse_rate <= 100:
            return 0  # Normal
        elif 101 < pulse_rate <= 110:
            return 1  # Mildly elevated
        elif 41 < pulse_rate <= 50:
            return 1  # Mildly elevated
        elif 111 < pulse_rate <= 129:
            return 2  # Elevated
        elif pulse_rate <= 40:
            return 2  # Elevated
        elif pulse_rate >= 130:
            return 3  # High
        else:
            return 3  # Very high

    def assign_mews_score_to_spo2(spo2):
        if 95 <= spo2:
            return 0  # Normal
        elif 90 <= spo2 <= 94:
            return 1  # Mildly reduced
        elif 86 <= spo2 <= 89:
            return 2  # Moderately reduced
        elif spo2 <= 85:
            return 3  # Severely reduced
        else:
            return 3  # Very severely reduced

    def assign_mews_score_to_respiratory_rate(respiratory_rate):
        if 9 <= respiratory_rate <= 14:
            return 0  # Normal
        elif 15 <= respiratory_rate <= 20:
            return 1  # Mildly elevated
        elif 21 <= respiratory_rate <= 29:
            return 2  # Elevated
        elif 30 <= respiratory_rate:
            return 3  # High
        else:
            return 3  # Very high

    data["PR_Val"] = data["PR_Val"].apply(assign_mews_score_to_pulse_rate)
    data["Resp_Val"] = data["Resp_Val"].apply(assign_mews_score_to_respiratory_rate)
    data["SPO2"] = data["SPO2"].apply(assign_mews_score_to_spo2)

# Step 4: Feature Calculation
def calculate_features(pmews_array):
    all_window_patterns = []
    for window in pmews_array:
        window['pattern'] = window.apply(lambda row: ''.join(map(str, row[1:])), axis=1)
        
        unique_patterns_in_window = {} 
        all_patterns = set(window['pattern']) - {'N'}  # Exclude missing values ('N')
        for pattern in all_patterns:
            pattern_data = window[window['pattern'] == pattern]
            m = len(pattern_data)

            if m >= 1:  # Ensure there are at least 2 data points to calculate slope
                time_intervals = np.array_split(window['Time'], 2)
                pattern_frequencies = []
                for interval in time_intervals:
                    pattern_frequencies.append(pattern_data[pattern_data['Time'].isin(interval)].shape[0])

                # Calculate slope
                x = [max(interval) for interval in time_intervals]
                y = pattern_frequencies
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                slope = numerator / denominator if denominator != 0 else 0    

                # Frequency counter for the pattern
                frequency = len(pattern_data)

                # Calculate trust
                trust = ((len(pattern) - pattern.count('N')) / len(pattern)) * 100

                # Calculate trend
                current_times = window['Time'].astype(int)
                first_occurrence = pattern_data.index[0]
                last_occurrence = pattern_data.index[-1]
                trend = (current_times[last_occurrence] - current_times[first_occurrence]) / frequency

                unique_patterns_in_window[pattern] = [trust, frequency, trend, slope]

        all_window_patterns.append(unique_patterns_in_window)

    return all_window_patterns 

# Step 5: Prioritization of Patterns
def prioritize_patterns(all_window_patterns):
    prioritized_patterns = []
    for window_patterns in all_window_patterns:
        patterns_with_priority = {}
        patterns = list(window_patterns.keys())

        # Calculate mean trust, trend, and frequency
        mean_trust = np.mean([window_patterns[p][0] for p in patterns])
        mean_trend = np.mean([window_patterns[p][2] for p in patterns])
        mean_frequency = np.mean([window_patterns[p][1] for p in patterns])

        for pattern, values in window_patterns.items():
            mews_score = sum(int(char) if char != 'N' else 0 for char in pattern)
            if mews_score >= 4:
                count = 0
                trust, frequency, trend, slope = values

                if trust >= mean_trust:
                    count += 1
                if trend >= mean_trend:
                    count += 1
                if frequency >= mean_frequency:
                    count += 1
                if slope >= 0:
                    count += 1

                patterns_with_priority[pattern] = 5-count  # Instead of rank it is count, when count is 4, rank is 1 and so on

        prioritized_patterns.append(patterns_with_priority)

    return prioritized_patterns

# Step 5: Visualization and Prioritized Alerts
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_results(prioritized_patterns):
    num_windows = len(prioritized_patterns)
    
    # Check if there's only one window to plot
    if num_windows == 1:
        fig, ax = plt.subplots(figsize=(15, 5))  # Single subplot with custom width
        window_patterns = prioritized_patterns[0]
        
        patterns = list(window_patterns.keys())
        ranks = list(window_patterns.values())
        
        # Color map and bar colors
        color_map = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'white'}
        colors = [color_map[rank] for rank in ranks]

        # Create the bar plot
        bars = ax.bar(patterns, ranks, color=colors)
        ax.set_xlabel('Patterns')
        ax.set_ylabel('Rank/Severity')
        ax.set_title('Pattern Ranks/Severity')
        ax.set_ylim(0, 5)  # Rank ranges from 1 to 5

    else:
        # Multiple windows: create a grid layout
        cols = 3
        rows = math.ceil(num_windows / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(15 * cols, 5 * rows))  # Increase width for multiple subplots
        axs = axs.flatten()

        color_map = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'white'}

        for i, (ax, window_patterns) in enumerate(zip(axs, prioritized_patterns)):
            patterns = list(window_patterns.keys())
            ranks = list(window_patterns.values())

            colors = [color_map[rank] for rank in ranks]

            bars = ax.bar(patterns, ranks, color=colors)
            ax.set_xlabel('Patterns')
            ax.set_ylabel('Rank/Severity')
            ax.set_title(f'Window {i + 1} - Pattern Ranks/Severity')
            ax.set_ylim(0, 5)

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

    # Legend for both cases
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Rank 1 (Red)', markersize=10, markerfacecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Rank 2 (Orange)', markersize=10, markerfacecolor='orange'),
        Line2D([0], [0], marker='o', color='w', label='Rank 3 (Yellow)', markersize=10, markerfacecolor='yellow'),
        Line2D([0], [0], marker='o', color='w', label='Rank 4 (Green)', markersize=10, markerfacecolor='green')
    ]

    pattern_explanation = "Pattern Code Explanation:\n1st: Pulse Rate, 2nd: Temperature, 3rd: Systolic BP, 4th: Respiratory Rate, 5th: SPO2"
    fig.text(0.1, 0.01, pattern_explanation, ha='left', va='bottom', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
    plt.show()


# Step 6: Algorithm Implementation
class MonitoringAlgorithm:
    def __init__(self, data):
        self.data = data
 
    def run(self):
        pass

# Step 7: Testing and Validation
def test_implementation():

    preprocessed_data = pd.read_csv("test_datasheet3.csv")
    pmews_array = create_sliding_windows(preprocessed_data, window_length=preprocessed_data.shape[0]-1  )
    features = calculate_features(pmews_array)
    priorities = prioritize_patterns(features)

    for feature in features:
        print(feature)
        


    visualize_results(priorities)

if __name__ == "__main__":
    test_implementation()
