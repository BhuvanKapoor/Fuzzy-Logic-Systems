import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input variables
density = ctrl.Antecedent(np.arange(0, 11, 1), 'density')
speed = ctrl.Antecedent(np.arange(0, 51, 1), 'speed')

# Membership functions for input variables
density['low'] = fuzz.trimf(density.universe, [0, 0, 3])
density['medium'] = fuzz.trimf(density.universe, [2, 5, 7])
density['high'] = fuzz.trimf(density.universe, [6, 9, 10])

speed['slow'] = fuzz.trapmf(speed.universe, [0, 0, 10, 20])
speed['moderate'] = fuzz.trapmf(speed.universe, [5, 15, 35, 45]) 
speed['fast'] = fuzz.trapmf(speed.universe, [35, 45, 55, 65])


# Output variable
green_time = ctrl.Consequent(np.arange(0, 101, 1), 'green_time')

# Membership functions for output variable
green_time['short'] = fuzz.trimf(green_time.universe, [0, 10, 30])
green_time['medium'] = fuzz.trimf(green_time.universe, [20, 50, 70])
green_time['long'] = fuzz.trimf(green_time.universe, [60, 100, 100])

# Fuzzy rules
rule1 = ctrl.Rule(density['low'] & speed['slow'], green_time['long']) 
rule2 = ctrl.Rule(density['low'] & speed['moderate'], green_time['medium']) 
rule3 = ctrl.Rule(density['low'] & speed['fast'], green_time['short']) 
rule4 = ctrl.Rule(density['medium'] & speed['slow'], green_time['medium'])
rule5 = ctrl.Rule(density['medium'] & speed['moderate'], green_time['medium'])
rule6 = ctrl.Rule(density['medium'] & speed['fast'], green_time['short'])
rule7 = ctrl.Rule(density['high'] & speed['slow'], green_time['short'])
rule8 = ctrl.Rule(density['high'] & speed['moderate'], green_time['short'])
rule9 = ctrl.Rule(density['high'] & speed['fast'], green_time['short'])

# Fuzzy inference system
traffic_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
traffic_light = ctrl.ControlSystemSimulation(traffic_light_ctrl)

# Test the system
traffic_light.input['density'] = 3
traffic_light.input['speed'] = 70
traffic_light.compute()
green_time_group = 'short' if traffic_light.output['green_time'] <= 30 else 'medium' if traffic_light.output['green_time'] <= 70 else 'long'
print(f"Green time duration: {round(traffic_light.output['green_time'],2)} seconds, Group: {green_time_group}")
