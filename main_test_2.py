import pandas as pd
import numpy as np
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
def create_sliding_windows(data: pd.DataFrame, window_length: int, window_increment:int):
    print("hello")
    sliding_windows = [];   
    for i in range(0, len(data)-window_length, window_increment):
        sliding_windows.append(data[i:i+window_length])
    return sliding_windows

# Step 3: Patterned Modified Early Warning Score (PMEWS)
def calculate_mews(data: pd.DataFrame):
    def assign_mews_score_to_pulse_rate(pulse_rate):
        # Define MEWS score ranges for Pulse Rate
        if 51<= pulse_rate <= 100:
            return 0  # Normal
        elif 101 < pulse_rate <= 110:
            return 1  # Mildly elevated
        elif 41 < pulse_rate <= 50:
            return 1  # Mildly elevated
        elif 111 < pulse_rate <= 129:
            return 2  # Elevated
        elif  pulse_rate <= 40:
            return 2  # Elevated
        elif pulse_rate >= 130:
            return 3  # High
        else:
            return 3  # Very high

    def assign_mews_score_to_spo2(spo2):
        # Define MEWS score ranges for Spo2
        if 95 <= spo2 :
            return 0  # Normal
        elif 90 <= spo2 <=94:
            return 1  # Mildly reduced
        elif 86 <= spo2 <= 89:
            return 2  # Moderately reduced
        elif  spo2 <= 85:
            return 3  # Severely reduced
        else:
            return 3  # Very severely reduced

    def assign_mews_score_to_respiratory_rate(respiratory_rate):
        # Define MEWS score ranges for Respiratory Rate
        if 9 <= respiratory_rate <= 14:
            return 0  # Normal
        elif 15 <= respiratory_rate <= 20:
            return 1  # Mildly elevated
        elif 21 <= respiratory_rate <= 29:
            return 2  # Elevated
        elif 30 <= respiratory_rate :
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
        calculate_mews(window)
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
                    
                    #print (max(interval))
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
                #current_patterns = window['pattern']
                current_times = window['Time'].astype(int)
                first_occurrence = pattern_data.index[0]
                last_occurrence = pattern_data.index[-1]
                trend = (current_times[last_occurrence] - current_times[first_occurrence]) / frequency

                unique_patterns_in_window[pattern] = [trust, frequency, trend,slope]

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
                rank = 0
                trust, frequency, trend, slope = values

                if trust >= mean_trust:
                    rank += 1
                if trend >= mean_trend:
                    rank += 1
                if frequency >= mean_frequency:
                    rank += 1
                if slope >= 0:
                    rank += 1

                patterns_with_priority[pattern] = rank

        prioritized_patterns.append(patterns_with_priority)

    return prioritized_patterns


# Step 5: Visualization and Prioritized Alerts
def visualize_results(all_window_patterns):
    # Assuming all_window_patterns is a list of dictionaries where each dictionary represents a window
    # and contains pattern frequencies and trust percentages.
    
    # Create a new figure for the plots
    pass


# Step 6: Algorithm Implementation
class MonitoringAlgorithm:
    def __init__(self, data):
        self.data = data

    def run(self):
        pass

# Step 7: Testing and Validation
def test_implementation():
    pass
    # ...

if __name__ == "__main__":
    raw_data = pd.read_csv("data.csv")
    preprocessed_data = preprocess_data(raw_data)
    pmews_array = create_sliding_windows(preprocessed_data, window_length=10, window_increment=5)
    features = calculate_features(pmews_array)
    priorities = prioritize_patterns(features)


    for feature in features:
        print(feature)
        
    for i, window in enumerate(priorities):
        print(f"Window {i+1}: {window}")    
    """
    visualize_results(features)
    algorithm = MonitoringAlgorithm(features)
    algorithm.run()
    test_implementation()
    """

