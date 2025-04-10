"""
تكامل مكونات نظام اكتشاف المواهب الرياضية

هذا الملف يقوم بدمج جميع مكونات النظام معًا واختبارها:
1. وحدة جمع البيانات
2. إطار تحليل اللاعبين
3. مكونات رؤية الحاسوب
4. نماذج التعلم الآلي
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# استيراد مكونات النظام
try:
    from data_collection import DataCollector, CSVDataSource, SensorDataSource, VideoDataSource
    from player_analysis import PlayerAnalyzer, PerformanceIndicator
    from computer_vision import PlayerDetector, PlayerTracker, PoseEstimator
    from machine_learning import PerformancePredictor, TalentIdentifier
except ImportError as e:
    print(f"خطأ في استيراد المكونات: {e}")
    print("سيتم استخدام نماذج بسيطة بدلاً من ذلك.")


class TalentScoutingSystem:
    """الفئة الرئيسية لنظام اكتشاف المواهب الرياضية"""
    
    def __init__(self, data_dir: str = "/tmp/sports_talent_data"):
        """
        تهيئة النظام
        
        المعلمات:
            data_dir: مجلد البيانات
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # تهيئة مكونات النظام
        self.data_collector = DataCollector()
        self.player_analyzer = PlayerAnalyzer(self.data_collector)
        self.video_analyzer = None  # سيتم تهيئته عند الحاجة
        
        # نماذج التعلم الآلي
        self.performance_predictor = PerformancePredictor()
        self.talent_identifier = TalentIdentifier()

        # بيانات النظام
        self.players_data = {}
        self.team_data = {}
        self.analysis_results = {}
    
    def add_data_sources(self, sources_config: List[Dict]) -> None:
        """
        إضافة مصادر بيانات متعددة
        
        المعلمات:
            sources_config: قائمة بإعدادات مصادر البيانات
        """
        for config in sources_config:
            source_type = config.get('type')
            
            if source_type == 'csv':
                source = CSVDataSource(config.get('path'), config.get('name'))
            elif source_type == 'sensor':
                source = SensorDataSource(config.get('id'), config.get('sensor_type'))
            elif source_type == 'video':
                source = VideoDataSource(config.get('path'), config.get('name'))
            else:
                print(f"نوع مصدر بيانات غير معروف: {source_type}")
                continue
            
            self.data_collector.add_data_source(source)
    
    def collect_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """
        جمع ومعالجة البيانات من جميع المصادر
        
        العائد:
            processed_data: البيانات المعالجة
        """
        # جمع البيانات
        collected_data = self.data_collector.collect_data()
        
        # معالجة البيانات
        processed_data = self.data_collector.preprocess_data(collected_data)
        
        # تخزين بيانات اللاعبين
        for player_id in range(1, 6):  # 5 لاعبين للتوضيح
            # تخزين بيانات GPS
            if "gps_player1_device" in processed_data:
                # في التطبيق الفعلي، سيتم تصفية البيانات حسب معرف اللاعب
                self.data_collector.store_player_data(player_id, processed_data["gps_player1_device"], "gps")
            
            # تخزين بيانات معدل ضربات القلب
            if "heart_rate_player1_device" in processed_data:
                self.data_collector.store_player_data(player_id, processed_data["heart_rate_player1_device"], "heart_rate")
            
            # تخزين بيانات الفيديو
            if "match_footage" in processed_data:
                # تصفية بيانات الفيديو حسب معرف اللاعب
                player_video_data = processed_data["match_footage"][
                    processed_data["match_footage"]["player_id"] == player_id
                ]
                self.data_collector.store_player_data(player_id, player_video_data, "video_tracking")
        
        return processed_data
    
    def analyze_players(self, player_ids: List[int]) -> Dict[int, Dict]:
        """
        تحليل مجموعة من اللاعبين
        
        المعلمات:
            player_ids: قائمة بمعرفات اللاعبين
            
        العائد:
            analysis_results: نتائج التحليل
        """
        analysis_results = {}
        
        for player_id in player_ids:
            # تحليل اللاعب
            player_report = self.player_analyzer.generate_player_report(player_id)
            analysis_results[player_id] = player_report
            
            # تخزين نتائج التحليل
            self.analysis_results[player_id] = player_report
        
        return analysis_results
    
    def analyze_video(self, video_path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> pd.DataFrame:
        """
        تحليل فيديو مباراة
        
        المعلمات:
            video_path: مسار ملف الفيديو
            start_frame: رقم الإطار البدائي
            end_frame: رقم الإطار النهائي (اختياري)
            
        العائد:
            video_analysis: نتائج تحليل الفيديو
        """
        # تهيئة محلل الفيديو
        self.video_analyzer = SoccerVideoAnalyzer(video_path)
        
        # تحميل الفيديو
        self.video_analyzer.load_video()
        
        # معالجة الفيديو
        video_analysis = self.video_analyzer.process_video(start_frame, end_frame)
        
        # تصدير نتائج التحليل
        export_path = os.path.join(self.data_dir, "video_analysis.csv")
        self.video_analyzer.export_analysis(export_path)
        
        # تمثيل مرئي لمسارات اللاعبين
        trajectories_path = os.path.join(self.data_dir, "player_trajectories.png")
        self.video_analyzer.visualize_trajectories(trajectories_path)
        
        return video_analysis
    
    def train_ml_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        تدريب نماذج التعلم الآلي
        
        المعلمات:
            training_data: بيانات التدريب
            
        العائد:
            model_metrics: مقاييس أداء النماذج
        """
        model_metrics = {}
        
        # 1. نموذج التنبؤ بالأداء
        performance_metrics = self.performance_predictor.train(
            training_data,
            ['overall_rating', 'market_value']
        )
        model_metrics['performance_predictor'] = performance_metrics
        
        # 2. نموذج اكتشاف المواهب
        self.talent_identifier.train(
            training_data,
            ['speed', 'endurance', 'agility', 'technical', 'tactical']
        )
        model_metrics['talent_identifier'] = {
            'thresholds': self.talent_identifier.get_talent_thresholds()
        }
        
        # 3. نموذج تجميع اللاعبين
        cluster_profiles = self.player_clusterer.train(
            training_data,
            ['speed', 'endurance', 'agility', 'technical', 'tactical']
        )
        model_metrics['player_clusterer'] = {
            'cluster_profiles': cluster_profiles
        }
        
        return model_metrics
    
    def predict_player_performance(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        التنبؤ بأداء لاعب
        
        المعلمات:
            player_data: بيانات اللاعب
            
        العائد:
            predictions: التنبؤات
        """
        return self.performance_predictor.predict(player_data)
    
    def identify_talents(self, players_data: pd.DataFrame) -> pd.DataFrame:
        """
        تحديد المواهب من بيانات اللاعبين
        
        المعلمات:
            players_data: بيانات اللاعبين
            
        العائد:
            talents: اللاعبون المصنفون كمواهب
        """
        return self.talent_identifier.identify_talents(players_data)
    
    def cluster_players(self, players_data: pd.DataFrame) -> pd.DataFrame:
        """
        تجميع اللاعبين حسب خصائصهم
        
        المعلمات:
            players_data: بيانات اللاعبين
            
        العائد:
            clustered_players: اللاعبون المجمعون
        """
        return self.player_clusterer.predict_cluster(players_data)
    
    def generate_training_recommendations(self, player_id: int) -> Dict[str, List[Dict]]:
        """
        توليد توصيات تدريبية مخصصة للاعب
        
        المعلمات:
            player_id: معرف اللاعب
            
        العائد:
            recommendations: التوصيات التدريبية
        """
        if player_id not in self.analysis_results:
            raise ValueError(f"لا توجد نتائج تحليل متاحة للاعب {player_id}. قم بتحليل اللاعب أولاً.")
        
        player_report = self.analysis_results[player_id]
        
        # استخراج البيانات اللازمة
        strengths = player_report['strengths']
        weaknesses = player_report['weaknesses']
        
        # اختيار المركز الأول الموصى به
        recommended_position = player_report['position_recommendations'][0]['position_name'] if player_report['position_recommendations'] else "لاعب وسط"
        
        # توليد التوصيات
        recommendations = self.training_recommender.generate_recommendations(
            {},  # بيانات اللاعب غير مستخدمة حاليًا
            strengths,
            weaknesses,
            recommended_position
        )
        
        return recommendations
    
    def create_training_plan(self, player_id: int, days_per_week: int = 5) -> List[Dict]:
        """
        إنشاء خطة تدريبية أسبوعية للاعب
        
        المعلمات:
            player_id: معرف اللاعب
            days_per_week: عدد أيام التدريب في الأسبوع
            
        العائد:
            training_plan: خطة التدريب
        """
        # توليد التوصيات
        recommendations = self.generate_training_recommendations(player_id)
        
        # إنشاء خطة التدريب
        training_plan = self.training_recommender.create_training_plan(recommendations, days_per_week)
        
        return training_plan
    
    def generate_player_report(self, player_id: int, output_path: Optional[str] = None) -> Dict:
        """
        إنشاء تقرير شامل عن اللاعب
        
        المعلمات:
            player_id: معرف اللاعب
            output_path: مسار حفظ التقرير (اختياري)
            
        العائد:
            report: التقرير
        """
        if player_id not in self.analysis_results:
            # تحليل اللاعب إذا لم يكن قد تم تحليله من قبل
            player_report = self.player_analyzer.generate_player_report(player_id)
            self.analysis_results[player_id] = player_report
        else:
            player_report = self.analysis_results[player_id]
        
        # إنشاء تمثيل مرئي لملف اللاعب
        if output_path:
            profile_image_path = os.path.join(output_path, f"player_{player_id}_profile.png")
            self.player_analyzer.visualize_player_profile(player_id, profile_image_path)
        
        # إنشاء خطة تدريبية
        training_plan = self.create_training_plan(player_id)
        
        # إضافة الخطة التدريبية إلى التقرير
        player_report['training_plan'] = training_plan
        
        # حفظ التقرير إذا تم تحديد مسار
        if output_path:
            report_path = os.path.join(output_path, f"player_{player_id}_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(player_report, f, ensure_ascii=False, indent=4)
        
        return player_report
    
    def export_data(self, output_path: str, format: str = "csv") -> None:
        """
        تصدير البيانات المجمعة إلى ملفات
        
        المعلمات:
            output_path: مسار الإخراج
            format: صيغة الملفات ('csv' أو 'json')
        """
        self.data_collector.export_data(output_path, format)


def create_sample_data() -> pd.DataFrame:
    """
    إنشاء بيانات تجريبية للاختبار
    
    العائد:
        data: إطار البيانات
    """
    # إنشاء بيانات تجريبية
    np.random.seed(42)
    
    # إنشاء 100 لاعب بخصائص عشوائية
    n_players = 100
    
    player_ids = np.arange(1, n_players + 1)
    ages = np.random.randint(18, 35, n_players)
    heights = np.random.normal(180, 10, n_players)
    weights = np.random.normal(75, 8, n_players)
    
    # مؤشرات الأداء
    speed = np.random.normal(70, 15, n_players)
    endurance = np.random.normal(65, 12, n_players)
    agility = np.random.normal(68, 14, n_players)
    technical = np.random.normal(72, 18, n_players)
    tactical = np.random.normal(60, 20, n_players)
    
    # إنشاء إطار البيانات
    data = pd.DataFrame({
        'player_id': player_ids,
        'age': ages,
        'height': heights,
        'weight': weights,
        'speed': speed,
        'endurance': endurance,
        'agility': agility,
        'technical': technical,
        'tactical': tactical
    })
    
    # إضافة بعض المتغيرات الفئوية
    positions = np.random.choice(['goalkeeper', 'defender', 'midfielder', 'winger', 'striker'], n_players)
    teams = np.random.choice(['team_a', 'team_b', 'team_c'], n_players)
    
    data['position'] = positions
    data['team'] = teams
    
    # إضافة متغير هدف للتنبؤ (مثلاً: تقييم عام)
    data['overall_rating'] = 0.2 * data['speed'] + 0.15 * data['endurance'] + 0.15 * data['agility'] + 0.25 * data['technical'] + 0.25 * data['tactical'] + np.random.normal(0, 5, n_players)
    data['overall_rating'] = data['overall_rating'].clip(0, 100)
    
    # إضافة متغير هدف آخر (مثلاً: القيمة السوقية)
    data['market_value'] = 100000 + 10000 * data['overall_rating'] + 5000 * (30 - data['age']) + np.random.normal(0, 200000, n_players)
    data['market_value'] = data['market_value'].clip(50000, 10000000)
    
    return data