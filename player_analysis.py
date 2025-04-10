"""
إطار تحليل اللاعبين لنظام اكتشاف المواهب الرياضية

هذا الإطار مسؤول عن:
1. تحليل بيانات اللاعبين وتحديد مؤشرات الأداء الرئيسية
2. اكتشاف نقاط القوة والضعف لكل لاعب
3. تحديد المراكز المناسبة للاعبين بناءً على خصائصهم
4. توليد توصيات تدريبية مخصصة
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta

# استيراد وحدة جمع البيانات
# في التطبيق الفعلي، سيتم استيراد الوحدة الحقيقية
try:
    from data_collection import DataCollector
except ImportError:
    print("تعذر استيراد وحدة جمع البيانات. سيتم استخدام نموذج بسيط بدلاً من ذلك.")
    # تعريف نموذج بسيط لـ DataCollector في حالة عدم توفر الوحدة الأصلية
    class DataCollector:
        def __init__(self):
            self.player_data = {}
        
        def get_player_data(self, player_id, data_type=None):
            return self.player_data.get(player_id, {})


class PerformanceIndicator:
    """فئة تمثل مؤشر أداء رئيسي للاعب"""
    
    def __init__(self, name: str, description: str, min_value: float = 0.0, max_value: float = 100.0):
        self.name = name
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
    
    def normalize(self, value: float) -> float:
        """تطبيع قيمة المؤشر إلى نطاق [0, 1]"""
        if value < self.min_value:
            return 0.0
        elif value > self.max_value:
            return 1.0
        else:
            return (value - self.min_value) / (self.max_value - self.min_value)
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب قيمة المؤشر من بيانات اللاعب"""
        raise NotImplementedError("يجب تنفيذ هذه الدالة في الفئات الفرعية")


class SpeedIndicator(PerformanceIndicator):
    """مؤشر سرعة اللاعب"""
    
    def __init__(self):
        super().__init__(
            name="السرعة",
            description="متوسط السرعة القصوى للاعب خلال المباراة",
            min_value=0.0,
            max_value=10.0  # م/ث
        )
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب متوسط السرعة القصوى من بيانات GPS"""
        if 'gps' not in player_data:
            return 0.0
        
        gps_data = player_data['gps']
        if 'speed' not in gps_data.columns:
            return 0.0
        
        # حساب متوسط أعلى 10% من قيم السرعة
        top_speeds = gps_data['speed'].nlargest(int(len(gps_data) * 0.1))
        return top_speeds.mean() if not top_speeds.empty else 0.0


class EnduranceIndicator(PerformanceIndicator):
    """مؤشر التحمل للاعب"""
    
    def __init__(self):
        super().__init__(
            name="التحمل",
            description="قدرة اللاعب على الحفاظ على مستوى أداء عالٍ لفترات طويلة",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب مؤشر التحمل من بيانات معدل ضربات القلب والمسافة المقطوعة"""
        endurance_score = 0.0
        
        # حساب المسافة الإجمالية المقطوعة (إذا كانت متوفرة)
        if 'gps' in player_data:
            gps_data = player_data['gps']
            if 'latitude' in gps_data.columns and 'longitude' in gps_data.columns:
                # في التطبيق الفعلي، سيتم حساب المسافة الحقيقية باستخدام صيغة هافرسين
                # هنا نستخدم تقديرًا بسيطًا للتوضيح
                total_distance = len(gps_data) * 0.1  # افتراض أن كل نقطة تمثل 0.1 متر
                endurance_score += min(total_distance / 1000, 50)  # تقييم حتى 50 نقطة للمسافة
        
        # تحليل معدل ضربات القلب (إذا كان متوفرًا)
        if 'heart_rate' in player_data:
            hr_data = player_data['heart_rate']
            if 'heart_rate' in hr_data.columns:
                # حساب معدل التعافي (انخفاض معدل ضربات القلب بعد المجهود)
                # في التطبيق الفعلي، سيتم تحليل منحنى التعافي بشكل أكثر تعقيدًا
                hr_max = hr_data['heart_rate'].max()
                hr_avg = hr_data['heart_rate'].mean()
                recovery_ratio = 1 - (hr_avg / hr_max) if hr_max > 0 else 0
                endurance_score += recovery_ratio * 50  # تقييم حتى 50 نقطة للتعافي
        
        return min(endurance_score, 100.0)  # التأكد من أن النتيجة لا تتجاوز 100


class AgilityIndicator(PerformanceIndicator):
    """مؤشر الرشاقة والمرونة للاعب"""
    
    def __init__(self):
        super().__init__(
            name="الرشاقة",
            description="قدرة اللاعب على تغيير الاتجاه بسرعة والتحرك بمرونة",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب مؤشر الرشاقة من بيانات تتبع الحركة"""
        if 'video_tracking' not in player_data:
            return 0.0
        
        tracking_data = player_data['video_tracking']
        if 'x_pos' not in tracking_data.columns or 'y_pos' not in tracking_data.columns:
            return 0.0
        
        # حساب التغيرات في الاتجاه
        # في التطبيق الفعلي، سيتم استخدام خوارزميات أكثر تعقيدًا لتحليل الحركة
        
        # حساب المسافة بين النقاط المتتالية
        x_diff = tracking_data['x_pos'].diff()
        y_diff = tracking_data['y_pos'].diff()
        
        # حساب زوايا الحركة
        angles = np.arctan2(y_diff, x_diff)
        
        # حساب التغيرات في الزوايا (تغييرات الاتجاه)
        angle_changes = np.abs(angles.diff())
        
        # تجاهل القيم المفقودة
        angle_changes = angle_changes.dropna()
        
        # حساب متوسط التغيرات في الاتجاه
        if len(angle_changes) > 0:
            avg_direction_change = angle_changes.mean() / np.pi  # تطبيع إلى [0, 1]
            return avg_direction_change * 100
        else:
            return 0.0


class TechnicalSkillIndicator(PerformanceIndicator):
    """مؤشر المهارات الفنية للاعب"""
    
    def __init__(self):
        super().__init__(
            name="المهارات الفنية",
            description="تقييم المهارات الفنية للاعب مثل التمرير والتسديد والمراوغة",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب مؤشر المهارات الفنية من بيانات الإحصائيات"""
        if 'statistics' not in player_data:
            return 50.0  # قيمة افتراضية في حالة عدم توفر البيانات
        
        stats = player_data['statistics']
        
        # في التطبيق الفعلي، سيتم استخدام إحصائيات حقيقية
        # هنا نفترض وجود أعمدة محددة في البيانات
        
        technical_score = 50.0  # قيمة أساسية
        
        # تقييم دقة التمرير (إذا كانت متوفرة)
        if 'passing_accuracy' in stats.columns:
            passing_score = stats['passing_accuracy'].mean() * 0.3  # 30% من التقييم
            technical_score += passing_score
        
        # تقييم دقة التسديد (إذا كانت متوفرة)
        if 'shooting_accuracy' in stats.columns:
            shooting_score = stats['shooting_accuracy'].mean() * 0.3  # 30% من التقييم
            technical_score += shooting_score
        
        # تقييم المراوغات الناجحة (إذا كانت متوفرة)
        if 'successful_dribbles' in stats.columns and 'dribble_attempts' in stats.columns:
            dribble_attempts = stats['dribble_attempts'].sum()
            if dribble_attempts > 0:
                dribble_success_rate = stats['successful_dribbles'].sum() / dribble_attempts
                dribble_score = dribble_success_rate * 0.2  # 20% من التقييم
                technical_score += dribble_score * 100
        
        # تقييم الاستحواذ على الكرة (إذا كان متوفرًا)
        if 'ball_possession' in stats.columns:
            possession_score = stats['ball_possession'].mean() * 0.2  # 20% من التقييم
            technical_score += possession_score
        
        return min(technical_score, 100.0)  # التأكد من أن النتيجة لا تتجاوز 100


class TacticalAwarenessIndicator(PerformanceIndicator):
    """مؤشر الوعي التكتيكي للاعب"""
    
    def __init__(self):
        super().__init__(
            name="الوعي التكتيكي",
            description="قدرة اللاعب على قراءة اللعب واتخاذ القرارات التكتيكية الصحيحة",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """حساب مؤشر الوعي التكتيكي من بيانات تتبع الحركة والإحصائيات"""
        tactical_score = 50.0  # قيمة أساسية
        
        # تحليل مواقع اللاعب على الملعب (إذا كانت متوفرة)
        if 'video_tracking' in player_data:
            tracking_data = player_data['video_tracking']
            if 'x_pos' in tracking_data.columns and 'y_pos' in tracking_data.columns:
                # في التطبيق الفعلي، سيتم تحليل مواقع اللاعب بالنسبة لزملائه والخصوم
                # هنا نستخدم تقديرًا بسيطًا للتوضيح
                
                # حساب تغطية الملعب (مساحة الحركة)
                x_range = tracking_data['x_pos'].max() - tracking_data['x_pos'].min()
                y_range = tracking_data['y_pos'].max() - tracking_data['y_pos'].min()
                field_coverage = (x_range * y_range) / (100 * 50)  # بافتراض أن أبعاد الملعب 100×50
                tactical_score += field_coverage * 25  # تقييم حتى 25 نقطة لتغطية الملعب
        
        # تحليل الإحصائيات التكتيكية (إذا كانت متوفرة)
        if 'statistics' in player_data:
            stats = player_data['statistics']
            
            # تقييم الاعتراضات والقطع (إذا كانت متوفرة)
            if 'interceptions' in stats.columns:
                interception_score = min(stats['interceptions'].mean() * 5, 15)  # تقييم حتى 15 نقطة
                tactical_score += interception_score
            
            # تقييم التمريرات المفتاحية (إذا كانت متوفرة)
            if 'key_passes' in stats.columns:
                key_pass_score = min(stats['key_passes'].mean() * 10, 15)  # تقييم حتى 15 نقطة
                tactical_score += key_pass_score
            
            # تقييم المساهمة الدفاعية (إذا كانت متوفرة)
            if 'tackles_won' in stats.columns and 'tackle_attempts' in stats.columns:
                tackle_attempts = stats['tackle_attempts'].sum()
                if tackle_attempts > 0:
                    tackle_success_rate = stats['tackles_won'].sum() / tackle_attempts
                    tackle_score = tackle_success_rate * 15  # تقييم حتى 15 نقطة
                    tactical_score += tackle_score
        
        return min(tactical_score, 100.0)  # التأكد من أن النتيجة لا تتجاوز 100


class PlayerAnalyzer:
    """فئة رئيسية لتحليل أداء اللاعبين وتحديد نقاط القوة والضعف"""
    
    def __init__(self, data_collector: DataCollector = None):
        self.data_collector = data_collector or DataCollector()
        
        # تعريف مؤشرات الأداء الرئيسية
        self.indicators = {
            'speed': SpeedIndicator(),
            'endurance': EnduranceIndicator(),
            'agility': AgilityIndicator(),
            'technical': TechnicalSkillIndicator(),
            'tactical': TacticalAwarenessIndicator()
        }
        
        # تعريف المراكز وخصائصها المطلوبة
        self.positions = {
            'goalkeeper': {
                'name': 'حارس مرمى',
                'key_indicators': ['agility', 'tactical'],
                'weights': {'speed': 0.1, 'endurance': 0.2, 'agility': 0.3, 'technical': 0.1, 'tactical': 0.3}
            },
            'defender': {
                'name': 'مدافع',
                'key_indicators': ['endurance', 'tactical'],
                'weights': {'speed': 0.2, 'endurance': 0.3, 'agility': 0.1, 'technical': 0.1, 'tactical': 0.3}
            },
            'midfielder': {
                'name': 'لاعب وسط',
                'key_indicators': ['endurance', 'technical', 'tactical'],
                'weights': {'speed': 0.2, 'endurance': 0.3, 'agility': 0.1, 'technical': 0.2, 'tactical': 0.2}
            },
            'winger': {
                'name': 'جناح',
                'key_indicators': ['speed', 'agility', 'technical'],
                'weights': {'speed': 0.3, 'endurance': 0.2, 'agility': 0.2, 'technical': 0.2, 'tactical': 0.1}
            },
            'striker': {
                'name': 'مهاجم',
                'key_indicators': ['speed', 'technical'],
                'weights': {'speed': 0.3, 'endurance': 0.1, 'agility': 0.2, 'technical': 0.3, 'tactical': 0.1}
            }
        }
    
    def analyze_player(self, player_id: int) -> Dict[str, float]:
        """تحليل أداء لاعب محدد وحساب مؤشراته"""
        # الحصول على بيانات اللاعب
        player_data = self.data_collector.get_player_data(player_id)
        
        # حساب قيم المؤشرات
        indicator_values = {}
        for indicator_id, indicator in self.indicators.items():
            try:
                value = indicator.calculate(player_data)
                indicator_values[indicator_id] = value
            except Exception as e:
                print(f"خطأ في حساب مؤشر {indicator_id} للاعب {player_id}: {e}")
                indicator_values[indicator_id] = 0.0
        
        return indicator_values
    
    def identify_strengths_weaknesses(self, indicator_values: Dict[str, float]) -> Dict[str, List[str]]:
        """تحديد نقاط القوة والضعف بناءً على قيم المؤشرات"""
        strengths = []
        weaknesses = []
        
        # تحديد المتوسط العام للمؤشرات
        avg_value = sum(indicator_values.values()) / len(indicator_values)
        
        # تحديد نقاط القوة (المؤشرات التي تتجاوز المتوسط بنسبة 20%)
        for indicator_id, value in indicator_values.items():
            indicator_name = self.indicators[indicator_id].name
            if value >= avg_value * 1.2:
                strengths.append(indicator_name)
            elif value <= avg_value * 0.8:
                weaknesses.append(indicator_name)
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def recommend_positions(self, indicator_values: Dict[str, float], top_n: int = 2) -> List[Dict[str, Union[str, float]]]:
        """توصية بأفضل المراكز المناسبة للاعب بناءً على مؤشراته"""
        position_scores = {}
        
        # حساب درجة التوافق مع كل مركز
        for position_id, position_info in self.positions.items():
            score = 0.0
            
            # حساب الدرجة المرجحة
            for indicator_id, weight in position_info['weights'].items():
                if indicator_id in indicator_values:
                    # تطبيع القيمة إلى نطاق [0, 1]
                    normalized_value = indicator_values[indicator_id] / 100.0
                    score += normalized_value * weight
            
            position_scores[position_id] = score
        
        # ترتيب المراكز حسب درجة التوافق
        sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
        
        # إعداد قائمة التوصيات
        recommendations = []
        for position_id, score in sorted_positions[:top_n]:
            position_info = self.positions[position_id]
            recommendations.append({
                'position_id': position_id,
                'position_name': position_info['name'],
                'compatibility_score': score,
                'compatibility_percentage': round(score * 100, 1)
            })
        
        return recommendations