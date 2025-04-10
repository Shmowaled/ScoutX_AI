import matplotlib.pyplot as plt
import numpy as np
from player_analysis import PlayerAnalyzer
from data_collection import DataCollector
from integrated_system import create_sample_data

# إنشاء بيانات تجريبية
sample_data = create_sample_data()

# إنشاء محلل اللاعبين
collector = DataCollector()
analyzer = PlayerAnalyzer(collector)

# تحليل لاعب
player_id = 1
indicator_values = {
    'speed': 85.0,
    'endurance': 70.0,
    'agility': 75.0,
    'technical': 80.0,
    'tactical': 65.0
}

# رسم مخطط راداري للاعب
categories = list(indicator_values.keys())
values = list(indicator_values.values())
categories.append(categories[0])
values.append(values[0])

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(np.linspace(0, 2*np.pi, len(categories)), values, 'o-', linewidth=2)
ax.fill(np.linspace(0, 2*np.pi, len(categories)), values, alpha=0.25)
ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(categories)-1)), categories[:-1])
ax.set_title(f"Player performance {player_id}")
plt.savefig("player_profile.png")
plt.show()
