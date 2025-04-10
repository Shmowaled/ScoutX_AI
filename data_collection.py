"""
وحدة جمع البيانات لنظام اكتشاف المواهب الرياضية

هذه الوحدة مسؤولة عن:
1. استيراد البيانات من مصادر مختلفة (ملفات CSV، قواعد بيانات، أجهزة استشعار)
2. معالجة البيانات الأولية وتنظيفها
3. تخزين البيانات في هيكل موحد للتحليل
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple

class DataSource:
    """فئة أساسية لمصادر البيانات المختلفة"""
    
    def __init__(self, source_name: str, source_type: str):
        self.source_name = source_name
        self.source_type = source_type
        self.last_update = None
    
    def connect(self) -> bool:
        """إنشاء اتصال بمصدر البيانات"""
        self.last_update = datetime.datetime.now()
        return True
    
    def disconnect(self) -> bool:
        """قطع الاتصال بمصدر البيانات"""
        return True
    
    def get_data(self) -> pd.DataFrame:
        """الحصول على البيانات من المصدر"""
        raise NotImplementedError("يجب تنفيذ هذه الدالة في الفئات الفرعية")


class CSVDataSource(DataSource):
    """مصدر بيانات من ملفات CSV"""
    
    def __init__(self, file_path: str, source_name: str = None):
        source_name = source_name or os.path.basename(file_path)
        super().__init__(source_name, "csv")
        self.file_path = file_path
    
    def get_data(self) -> pd.DataFrame:
        """قراءة البيانات من ملف CSV"""
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            print(f"خطأ في قراءة ملف CSV: {e}")
            return pd.DataFrame()


class SensorDataSource(DataSource):
    """مصدر بيانات من أجهزة الاستشعار (GPS، Bluetooth)"""
    
    def __init__(self, sensor_id: str, sensor_type: str):
        super().__init__(f"{sensor_type}_{sensor_id}", "sensor")
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        
    def get_data(self) -> pd.DataFrame:
        """
        الحصول على البيانات من جهاز الاستشعار
        
        ملاحظة: هذه دالة تمثيلية فقط. في التطبيق الفعلي، ستتصل بأجهزة الاستشعار
        عبر واجهات برمجة التطبيقات المناسبة.
        """
        # في التطبيق الفعلي، سيتم استبدال هذا بالاتصال الحقيقي بأجهزة الاستشعار
        # مثال: استخدام مكتبات مثل bluepy للاتصال بأجهزة Bluetooth
        
        # إنشاء بيانات تجريبية للتوضيح
        timestamps = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(minutes=5),
                                  end=datetime.datetime.now(),
                                  periods=100)
        
        if self.sensor_type == "gps":
            # بيانات GPS تجريبية (الوقت، خط الطول، خط العرض، السرعة)
            data = {
                'timestamp': timestamps,
                'latitude': np.random.uniform(24.0, 25.0, 100),  # مثال: إحداثيات في الرياض
                'longitude': np.random.uniform(46.5, 47.5, 100),
                'speed': np.random.uniform(0, 8, 100)  # السرعة بالمتر/ثانية
            }
        elif self.sensor_type == "heart_rate":
            # بيانات معدل ضربات القلب تجريبية
            data = {
                'timestamp': timestamps,
                'heart_rate': np.random.normal(75, 15, 100)  # معدل ضربات القلب النموذجي أثناء التمرين
            }
        elif self.sensor_type == "acceleration":
            # بيانات التسارع تجريبية
            data = {
                'timestamp': timestamps,
                'acc_x': np.random.normal(0, 2, 100),
                'acc_y': np.random.normal(0, 2, 100),
                'acc_z': np.random.normal(9.8, 1, 100)  # تسارع الجاذبية + تغيرات
            }
        else:
            # بيانات افتراضية
            data = {
                'timestamp': timestamps,
                'value': np.random.random(100)
            }
            
        return pd.DataFrame(data)


class VideoDataSource(DataSource):
    """
    مصدر بيانات من ملفات الفيديو
    
    ملاحظة: هذه فئة تمثيلية فقط. في التطبيق الفعلي، ستستخدم مكتبات مثل OpenCV و YOLO
    لاستخراج البيانات من الفيديو.
    """
    
    def __init__(self, video_path: str, source_name: str = None):
        source_name = source_name or os.path.basename(video_path)
        super().__init__(source_name, "video")
        self.video_path = video_path
        
    def get_data(self) -> pd.DataFrame:
        """
        استخراج البيانات من ملف الفيديو
        
        ملاحظة: في التطبيق الفعلي، سيتم استخدام OpenCV و YOLO لتحليل الفيديو
        واستخراج مواقع اللاعبين وحركاتهم.
        """
        # إنشاء بيانات تجريبية للتوضيح
        # في التطبيق الفعلي، سيتم استبدال هذا بتحليل الفيديو الحقيقي
        
        # تمثيل 10 إطارات من الفيديو مع 5 لاعبين في كل إطار
        frames = []
        for frame_id in range(10):
            for player_id in range(1, 6):
                frames.append({
                    'frame_id': frame_id,
                    'timestamp': datetime.datetime.now() + datetime.timedelta(seconds=frame_id/30),  # بافتراض 30 إطار/ثانية
                    'player_id': player_id,
                    'x_pos': np.random.uniform(0, 100),  # الموقع الأفقي على الملعب
                    'y_pos': np.random.uniform(0, 50),   # الموقع الرأسي على الملعب
                    'confidence': np.random.uniform(0.7, 1.0)  # ثقة الكشف
                })
        
        return pd.DataFrame(frames)


class DataCollector:
    """فئة رئيسية لجمع البيانات من مصادر متعددة وتوحيدها"""
    
    def __init__(self):
        self.data_sources = {}
        self.player_data = {}
        self.team_data = {}
        
    def add_data_source(self, source: DataSource) -> None:
        """إضافة مصدر بيانات جديد"""
        self.data_sources[source.source_name] = source
        print(f"تمت إضافة مصدر البيانات: {source.source_name} من النوع {source.source_type}")
    
    def collect_data(self, source_name: str = None) -> Dict[str, pd.DataFrame]:
        """جمع البيانات من مصدر محدد أو من جميع المصادر"""
        collected_data = {}
        
        if source_name and source_name in self.data_sources:
            # جمع البيانات من مصدر محدد
            source = self.data_sources[source_name]
            collected_data[source_name] = source.get_data()
        else:
            # جمع البيانات من جميع المصادر
            for name, source in self.data_sources.items():
                try:
                    data = source.get_data()
                    collected_data[name] = data
                    print(f"تم جمع {len(data)} سجل من المصدر {name}")
                except Exception as e:
                    print(f"خطأ في جمع البيانات من المصدر {name}: {e}")
        
        return collected_data
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """معالجة البيانات الأولية وتنظيفها"""
        processed_data = {}
        
        for source_name, df in data.items():
            # التعامل مع القيم المفقودة
            df_clean = df.copy()
            df_clean = df_clean.dropna()  # حذف الصفوف ذات القيم المفقودة
            
            # تنسيق أعمدة التاريخ والوقت
            if 'timestamp' in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
                df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            
            # إضافة معالجة خاصة حسب نوع المصدر
            source = self.data_sources.get(source_name)
            if source and source.source_type == "sensor":
                # معالجة بيانات أجهزة الاستشعار (مثل تنعيم البيانات)
                if 'value' in df_clean.columns:
                    # تطبيق المتوسط المتحرك لتنعيم البيانات
                    df_clean['value_smooth'] = df_clean['value'].rolling(window=5, min_periods=1).mean()
            
            processed_data[source_name] = df_clean
            
        return processed_data
    
    def store_player_data(self, player_id: int, data: pd.DataFrame, data_type: str) -> None:
        """تخزين بيانات اللاعب"""
        if player_id not in self.player_data:
            self.player_data[player_id] = {}
        
        self.player_data[player_id][data_type] = data
        print(f"تم تخزين بيانات {data_type} للاعب {player_id}")
    
    def get_player_data(self, player_id: int, data_type: str = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """استرجاع بيانات اللاعب"""
        if player_id not in self.player_data:
            print(f"لا توجد بيانات متاحة للاعب {player_id}")
            return pd.DataFrame() if data_type else {}
        
        if data_type:
            return self.player_data[player_id].get(data_type, pd.DataFrame())
        else:
            return self.player_data[player_id]
    
    def export_data(self, output_path: str, format: str = "csv") -> None:
        """تصدير البيانات المجمعة إلى ملفات"""
        os.makedirs(output_path, exist_ok=True)
        
        # تصدير بيانات اللاعبين
        for player_id, player_data_dict in self.player_data.items():
            player_dir = os.path.join(output_path, f"player_{player_id}")
            os.makedirs(player_dir, exist_ok=True)
            
            for data_type, data in player_data_dict.items():
                if format == "csv":
                    file_path = os.path.join(player_dir, f"{data_type}.csv")
                    data.to_csv(file_path, index=False)
                elif format == "json":
                    file_path = os.path.join(player_dir, f"{data_type}.json")
                    data.to_json(file_path, orient="records")
                
                print(f"تم تصدير بيانات {data_type} للاعب {player_id} إلى {file_path}")


# مثال على استخدام وحدة جمع البيانات
def demo_data_collection():
    """عرض توضيحي لاستخدام وحدة جمع البيانات"""
    # إنشاء مجمع البيانات
    collector = DataCollector()
    
    # إضافة مصادر بيانات متنوعة (تمثيلية)
    # في التطبيق الفعلي، ستكون هذه مصادر بيانات حقيقية
    
    # مصدر بيانات CSV (بيانات إحصائية للاعبين)
    # ملاحظة: هذا مسار تمثيلي، سيتم استبداله بمسار حقيقي في التطبيق الفعلي
    csv_source = CSVDataSource("/path/to/player_stats.csv", "player_statistics")
    collector.add_data_source(csv_source)
    
    # مصادر بيانات أجهزة الاستشعار
    gps_source = SensorDataSource("player1_device", "gps")
    heart_rate_source = SensorDataSource("player1_device", "heart_rate")
    collector.add_data_source(gps_source)
    collector.add_data_source(heart_rate_source)
    
    # مصدر بيانات فيديو
    video_source = VideoDataSource("/path/to/match_video.mp4", "match_footage")
    collector.add_data_source(video_source)
    
    # جمع البيانات من جميع المصادر
    collected_data = collector.collect_data()
    
    # معالجة البيانات
    processed_data = collector.preprocess_data(collected_data)
    
    # تخزين بيانات اللاعبين
    for player_id in range(1, 6):  # 5 لاعبين للتوضيح
        # تخزين بيانات GPS
        if "gps_player1_device" in processed_data:
            # في التطبيق الفعلي، سيتم تصفية البيانات حسب معرف اللاعب
            collector.store_player_data(player_id, processed_data["gps_player1_device"], "gps")
        
        # تخزين بيانات معدل ضربات القلب
        if "heart_rate_player1_device" in processed_data:
            collector.store_player_data(player_id, processed_data["heart_rate_player1_device"], "heart_rate")
        
        # تخزين بيانات الفيديو
        if "match_footage" in processed_data:
            # تصفية بيانات الفيديو حسب معرف اللاعب
            player_video_data = processed_data["match_footage"][
                processed_data["match_footage"]["player_id"] == player_id
            ]
            collector.store_player_data(player_id, player_video_data, "video_tracking")
    
    # تصدير البيانات
    collector.export_data("/path/to/output", "csv")
    
    return collector


if __name__ == "__main__":
    # تشغيل العرض التوضيحي
    collector = demo_data_collection()
    
    # عرض بيانات لاعب معين
    player_id = 1
    player_data = collector.get_player_data(player_id)
    
    for data_type, data in player_data.items():
        print(f"\nبيانات {data_type} للاعب {player_id}:")
        print(data.head())