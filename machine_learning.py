"""
نماذج التعلم الآلي لنظام اكتشاف المواهب الرياضية

هذه الوحدة مسؤولة عن:
1. بناء نماذج للتنبؤ بأداء اللاعبين المستقبلي
2. تطوير خوارزميات لاكتشاف المواهب الواعدة
3. إنشاء نظام توصيات للتدريب الشخصي
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# استيراد وحدات النظام الأخرى
# في التطبيق الفعلي، سيتم استيراد الوحدات الحقيقية
try:
    from data_collection import DataCollector
    from player_analysis import PlayerAnalyzer
except ImportError:
    print("تعذر استيراد بعض وحدات النظام. سيتم استخدام نماذج بسيطة بدلاً من ذلك.")


class DataPreprocessor:
    """فئة لتجهيز البيانات للنماذج"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
    
    def prepare_performance_data(self, data: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        تجهيز بيانات الأداء للنماذج التنبؤية
        
        المعلمات:
            data: إطار البيانات الأصلي
            target_columns: أعمدة الهدف للتنبؤ
            
        العائد:
            X: ميزات النموذج
            y: أهداف النموذج
        """
        # نسخ البيانات لتجنب التعديل على الأصل
        df = data.copy()
        
        # التعامل مع القيم المفقودة
        df = df.dropna()
        
        # فصل الميزات والأهداف
        y = df[target_columns]
        X = df.drop(columns=target_columns)
        
        # حفظ أسماء الأعمدة
        self.feature_columns = X.columns.tolist()
        self.target_columns = target_columns
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """
        تطبيع قيم الميزات
        
        المعلمات:
            X: ميزات النموذج
            scaler_type: نوع التطبيع ('standard' أو 'minmax')
            
        العائد:
            X_scaled: الميزات المطبعة
        """
        X_scaled = X.copy()
        
        # إنشاء مطبع جديد
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"نوع تطبيع غير مدعوم: {scaler_type}")
        
        # تطبيق التطبيع على الأعمدة العددية فقط
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['features'] = scaler
        
        return X_scaled
    
    def encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ترميز المتغيرات الفئوية
        
        المعلمات:
            X: ميزات النموذج
            
        العائد:
            X_encoded: الميزات المرمزة
        """
        # نسخ البيانات
        X_encoded = X.copy()
        
        # ترميز المتغيرات الفئوية باستخدام one-hot encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        return X_encoded
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        تقليل أبعاد الميزات باستخدام PCA
        
        المعلمات:
            X: ميزات النموذج
            n_components: عدد المكونات المطلوبة
            
        العائد:
            X_reduced: الميزات ذات الأبعاد المخفضة
        """
        # التأكد من أن عدد المكونات لا يتجاوز عدد الأعمدة
        n_components = min(n_components, X.shape[1])
        
        # تطبيق PCA
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # تحويل المصفوفة إلى إطار بيانات
        X_reduced = pd.DataFrame(
            X_reduced,
            index=X.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # حفظ نموذج PCA
        self.scalers['pca'] = pca
        
        return X_reduced
    
    def prepare_pipeline(self, data: pd.DataFrame, target_columns: List[str], use_pca: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        تنفيذ سلسلة كاملة من عمليات تجهيز البيانات
        
        المعلمات:
            data: إطار البيانات الأصلي
            target_columns: أعمدة الهدف للتنبؤ
            use_pca: ما إذا كان سيتم استخدام PCA لتقليل الأبعاد
            
        العائد:
            X_processed: الميزات المجهزة
            y: أهداف النموذج
        """
        # تجهيز البيانات
        X, y = self.prepare_performance_data(data, target_columns)
        
        # ترميز المتغيرات الفئوية
        X_encoded = self.encode_categorical(X)
        
        # تطبيع الميزات
        X_scaled = self.scale_features(X_encoded)
        
        # تقليل الأبعاد (اختياري)
        if use_pca and X_scaled.shape[1] > 10:
            X_processed = self.reduce_dimensions(X_scaled)
        else:
            X_processed = X_scaled
        
        return X_processed, y


class PerformancePredictor:
    """فئة للتنبؤ بأداء اللاعبين المستقبلي"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        تهيئة متنبئ الأداء
        
        المعلمات:
            model_type: نوع النموذج ('random_forest', 'linear', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_importance = None
    
    def train(self, data: pd.DataFrame, target_columns: List[str], test_size: float = 0.2) -> Dict[str, float]:
        """
        تدريب نموذج التنبؤ
        
        المعلمات:
            data: إطار البيانات
            target_columns: أعمدة الهدف للتنبؤ
            test_size: نسبة بيانات الاختبار
            
        العائد:
            metrics: مقاييس أداء النموذج
        """
        # تجهيز البيانات
        X, y = self.preprocessor.prepare_pipeline(data, target_columns)
        
        # تقسيم البيانات إلى تدريب واختبار
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # إنشاء النموذج
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"نوع نموذج غير مدعوم: {self.model_type}")
        
        # تدريب النموذج
        self.model.fit(X_train, y_train)
        
        # حساب أهمية الميزات (إذا كان النموذج يدعم ذلك)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # تقييم النموذج
        y_pred = self.model.predict(X_test)
        
        # حساب مقاييس الأداء
        metrics = {}
        
        # إذا كان هناك هدف واحد فقط
        if len(target_columns) == 1:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, y_pred)
        else:
            # حساب المقاييس لكل هدف
            for i, col in enumerate(target_columns):
                metrics[f'{col}_mse'] = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
                metrics[f'{col}_rmse'] = np.sqrt(metrics[f'{col}_mse'])
                metrics[f'{col}_r2'] = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        return metrics
    
    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        التنبؤ بأداء لاعب
        
        المعلمات:
            player_data: بيانات اللاعب
            
        العائد:
            predictions: التنبؤات
        """
        if self.model is None:
            raise ValueError("لم يتم تدريب النموذج بعد. قم بتدريب النموذج أولاً باستخدام الدالة train().")
        
        # تجهيز البيانات
        X = player_data.copy()
        
        # التأكد من وجود جميع الأعمدة المطلوبة
        for col in self.preprocessor.feature_columns:
            if col not in X.columns:
                X[col] = 0  # قيمة افتراضية
        
        # ترتيب الأعمدة بنفس ترتيب التدريب
        X = X[self.preprocessor.feature_columns]
        
        # ترميز المتغيرات الفئوية
        X_encoded = self.preprocessor.encode_categorical(X)
        
        # تطبيع الميزات
        if 'features' in self.preprocessor.scalers:
            numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
            X_encoded[numeric_cols] = self.preprocessor.scalers['features'].transform(X_encoded[numeric_cols])
        
        # تقليل الأبعاد (إذا تم استخدام PCA)
        if 'pca' in self.preprocessor.scalers:
            X_processed = self.preprocessor.scalers['pca'].transform(X_encoded)
            X_processed = pd.DataFrame(
                X_processed,
                index=X_encoded.index,
                columns=[f'PC{i+1}' for i in range(X_processed.shape[1])]
            )
        else:
            X_processed = X_encoded
        
        # التنبؤ
        predictions = self.model.predict(X_processed)
        
        # تحويل التنبؤات إلى إطار بيانات
        if len(self.preprocessor.target_columns) == 1:
            predictions = pd.DataFrame(
                predictions,
                index=X.index,
                columns=self.preprocessor.target_columns
            )
        else:
            predictions = pd.DataFrame(
                predictions,
                index=X.index,
                columns=self.preprocessor.target_columns
            )
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        الحصول على أهم الميزات في النموذج
        
        المعلمات:
            top_n: عدد الميزات المطلوبة
            
        العائد:
            top_features: أهم الميزات
        """
        if self.feature_importance is None:
            return {}
        
        # ترتيب الميزات حسب الأهمية
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # اختيار أهم الميزات
        top_features = dict(sorted_features[:top_n])
        
        return top_features
    
    def save_model(self, model_path: str) -> None:
        """
        حفظ النموذج إلى ملف
        
        المعلمات:
            model_path: مسار الملف
        """
        if self.model is None:
            raise ValueError("لم يتم تدريب النموذج بعد. قم بتدريب النموذج أولاً باستخدام الدالة train().")
        
        # إنشاء مجلد الإخراج إذا لم يكن موجودًا
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # حفظ النموذج والمعالج
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, model_path: str) -> 'PerformancePredictor':
        """
        تحميل النموذج من ملف
        
        المعلمات:
            model_path: مسار الملف
            
        العائد:
            predictor: متنبئ الأداء
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # إنشاء متنبئ جديد
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.preprocessor = model_data['preprocessor']
        predictor.feature_importance = model_data['feature_importance']
        
        return predictor


class TalentIdentifier:
    """فئة لاكتشاف المواهب الواعدة"""
    
    def __init__(self, threshold_percentile: int = 90):
        """
        تهيئة محدد المواهب
        
        المعلمات:
            threshold_percentile: النسبة المئوية للعتبة (اللاعبون فوق هذه النسبة يعتبرون مواهب)
        """
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.talent_thresholds = {}
    
    def train(self, data: pd.DataFrame, performance_indicators: List[str]) -> None:
        """
        تدريب نموذج اكتشاف المواهب
        
        المعلمات:
            data: إطار البيانات
            performance_indicators: مؤشرات الأداء المستخدمة لتحديد المواهب
        """
        # نسخ البيانات
        df = data.copy()
        
        # التأكد من وجود جميع مؤشرات الأداء
        for indicator in performance_indicators:
            if indicator not in df.columns:
                raise ValueError(f"مؤشر الأداء غير موجود في البيانات: {indicator}")
        
        # حساب عتبات المواهب لكل مؤشر
        for indicator in performance_indicators:
            threshold = np.percentile(df[indicator], self.threshold_percentile)
            self.talent_thresholds[indicator] = threshold
        
        # إنشاء عمود جديد يشير إلى ما إذا كان اللاعب موهبة أم لا
        df['is_talent'] = 0
        
        for indicator in performance_indicators:
            df.loc[df[indicator] >= self.talent_thresholds[indicator], 'is_talent'] = 1
        
        # تجهيز البيانات
        X, y = self.preprocessor.prepare_pipeline(df, ['is_talent'])
        
        # تدريب نموذج التصنيف
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def identify_talents(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        تحديد المواهب من بيانات اللاعبين
        
        المعلمات:
            player_data: بيانات اللاعبين
            
        العائد:
            talents: اللاعبون المصنفون كمواهب
        """
        if self.model is None:
            raise ValueError("لم يتم تدريب النموذج بعد. قم بتدريب النموذج أولاً باستخدام الدالة train().")
        
        # نسخ البيانات
        df = player_data.copy()
        
        # تجهيز البيانات
        X = df.copy()
        
        # التأكد من وجود جميع الأعمدة المطلوبة
        for col in self.preprocessor.feature_columns:
            if col not in X.columns:
                X[col] = 0  # قيمة افتراضية
        
        # ترتيب الأعمدة بنفس ترتيب التدريب
        X = X[self.preprocessor.feature_columns]
        
        # ترميز المتغيرات الفئوية
        X_encoded = self.preprocessor.encode_categorical(X)
        
        # تطبيع الميزات
        if 'features' in self.preprocessor.scalers:
            numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
            X_encoded[numeric_cols] = self.preprocessor.scalers['features'].transform(X_encoded[numeric_cols])
        
        # تقليل الأبعاد (إذا تم استخدام PCA)
        if 'pca' in self.preprocessor.scalers:
            X_processed = self.preprocessor.scalers['pca'].transform(X_encoded)
            X_processed = pd.DataFrame(
                X_processed,
                index=X_encoded.index,
                columns=[f'PC{i+1}' for i in range(X_processed.shape[1])]
            )
        else:
            X_processed = X_encoded
        
        # التنبؤ
        talent_proba = self.model.predict_proba(X_processed)[:, 1]  # احتمالية أن يكون اللاعب موهبة